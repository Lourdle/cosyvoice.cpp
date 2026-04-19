#!/usr/bin/env python3
"""Generate gold for HiFT vocoder layer-by-layer comparison.

Loads all HiFT weights from GGUF, runs the full HiFT pipeline in PyTorch
on flow_output_final.bin (with zero noise and zero rand_ini), and saves
intermediate outputs at each stage.

Requires:
  - /tmp/cv_repro3/flow_output_final.bin from the voice clone reproduction
  - GGUF model file

Output: /tmp/cv_hift_gold/
"""
import math, os, struct, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import gguf

GGUF_PATH = os.environ.get(
    "COSYVOICE_TEST_MODEL",
    os.path.join(os.path.dirname(__file__), "../../models/tts/cosyvoice/CosyVoice3-2512_Q8_0.gguf"),
)
REPRO_DIR = "/tmp/cv_repro3"
OUT = "/tmp/cv_hift_gold"
os.makedirs(OUT, exist_ok=True)


def save_f32(name, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().float().numpy()
    flat = data.astype(np.float32).flatten()
    path = os.path.join(OUT, f"{name}.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(data.shape)))
        for d_ in data.shape:
            f.write(struct.pack("<q", d_))
        f.write(flat.tobytes())
    print(f"  {name}: shape={list(data.shape)}, numel={flat.size}")


def _dequant(t):
    ttype = t.tensor_type
    if ttype in (0, 1):
        return t.data.astype(np.float32).flatten()
    return gguf.dequantize(t.data, gguf.GGMLQuantizationType(ttype)).flatten()


class GGUFLoader:
    def __init__(self, path):
        self.reader = gguf.GGUFReader(path)
        self._cache = {t.name: t for t in self.reader.tensors}

    def tensor(self, name):
        t = self._cache[name]
        return _dequant(t), [int(x) for x in t.shape]

    def has(self, name):
        return name in self._cache

    def linear(self, prefix, in_d=None, out_d=None):
        w, w_ne = self.tensor(f"{prefix}.weight")
        b, _ = self.tensor(f"{prefix}.bias")
        if in_d is None:
            in_d = w_ne[0]
        if out_d is None:
            out_d = w_ne[1] if len(w_ne) > 1 else 1
        layer = nn.Linear(in_d, out_d)
        layer.weight.data = torch.from_numpy(w.reshape(out_d, in_d).copy())
        layer.bias.data = torch.from_numpy(b[:out_d].copy())
        return layer

    def conv1d(self, prefix):
        """Load Conv1d weight/bias from GGUF.
        GGUF ne=[kernel, in_ch, out_ch] → PyTorch [out_ch, in_ch, kernel]
        """
        w, w_ne = self.tensor(f"{prefix}.weight")
        b, _ = self.tensor(f"{prefix}.bias")
        kernel, in_ch, out_ch = w_ne[0], w_ne[1], w_ne[2]
        return (
            torch.from_numpy(w.reshape(out_ch, in_ch, kernel).copy()),
            torch.from_numpy(b[:out_ch].copy()),
            kernel, in_ch, out_ch
        )

    def get_arr(self, name):
        """Extract int32 array from GGUF metadata."""
        for f in self.reader.fields.values():
            if f.name == name:
                return [int(f.parts[idx][0]) for idx in f.data]
        return None

    def get_f32(self, name):
        for f in self.reader.fields.values():
            if f.name == name:
                return float(f.parts[-1][0])
        return None

    def get_i32(self, name):
        for f in self.reader.fields.values():
            if f.name == name:
                return int(f.parts[-1][0])
        return None


# --- Causal Conv1d ---

def causal_padding_size(kernel_size, dilation):
    """Match C++ CausalConv1d::causal_padding()"""
    return (kernel_size * dilation - dilation) // 2 * 2 + (kernel_size + 1) % 2


def causal_conv1d(x, weight, bias, causal_type, dilation=1):
    """Apply causal Conv1d matching C++ CausalConv1d::build_cgraph.
    x: [B, C, L]
    weight: [out_ch, in_ch, kernel]
    causal_type: 'left' or 'right'
    """
    pad = causal_padding_size(weight.shape[2], dilation)
    if causal_type == 'left':
        x = F_torch.pad(x, (pad, 0))
    else:  # right
        x = F_torch.pad(x, (0, pad))
    return F_torch.conv1d(x, weight, bias, dilation=dilation)


# --- Snake activation ---

def snake(x, alpha, eps=1e-9):
    """Snake activation: x + sin²(αx)/α"""
    alpha = alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
    alpha = torch.clamp(alpha.abs(), min=eps)
    x_sin = torch.sin(x * alpha)
    return x + x_sin * x_sin / alpha


# --- F0 Predictor ---

class F0Predictor:
    def __init__(self, loader):
        self.convs = []
        self.causal_types = ['right', 'left', 'left', 'left', 'left']
        for i, idx in enumerate([0, 2, 4, 6, 8]):
            w, b, k, ic, oc = loader.conv1d(f"f0_predictor.condnet.{idx}")
            self.convs.append((w, b))
        self.classifier = loader.linear("f0_predictor.classifier", in_d=512, out_d=1)

    @torch.no_grad()
    def forward(self, x):
        """x: [B, 80, L] → f0: [B, L]"""
        for i, ((w, b), ct) in enumerate(zip(self.convs, self.causal_types)):
            x = causal_conv1d(x, w, b, ct)
            x = F_torch.elu(x)
        # x: [B, 512, L] → permute to [B, L, 512] → classifier → [B, L, 1]
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        x = x.squeeze(-1)  # [B, L]
        return x.abs()


# --- SineGen2 (simplified: zero rand_ini) ---

@torch.no_grad()
def sinegen2(f0, harmonic_num, sampling_rate, upsample_scale):
    """Simplified SineGen2 with zero rand_ini and zero noise.
    f0: [B, L] (Hz)
    Returns: sine_waves [B, harmonic_num+1, L*upsample_scale]
    """
    B, L = f0.shape
    sine_amp = 0.1  # nsf_alpha

    # Create harmonic multipliers [1, 2, ..., harmonic_num+1]
    harmonics = torch.arange(1, harmonic_num + 2, dtype=torch.float32)  # [H+1]

    # f0 * harmonics: [B, H+1, L]
    f0_h = f0.unsqueeze(1) * harmonics.unsqueeze(0).unsqueeze(-1)  # [B, H+1, L]

    # Normalize: fraction of sample rate
    rad = f0_h / sampling_rate  # [B, H+1, L]
    rad = rad - torch.floor(rad)  # fractional part

    # rand_ini = zeros → no initial phase offset (we skip ggml_acc step)

    # Bilinear interpolation (downsample by upsample_scale, then cumsum, then upsample)
    # In C++: permute [H+1, L, B] → interpolate L → L/upsample_scale (bilinear)
    # For PyTorch: [B, H+1, L] → interpolate length dim
    down_len = max(1, L // upsample_scale)
    rad_down = F_torch.interpolate(rad, size=down_len, mode='linear', align_corners=False)

    # Cumulative sum for continuous phase
    phase = torch.cumsum(rad_down, dim=-1)
    phase = phase * (2.0 * math.pi * upsample_scale)

    # Upsample back to full length
    up_len = L * upsample_scale
    phase = F_torch.interpolate(phase, size=up_len, mode='nearest')

    # Generate sine waves
    sines = torch.sin(phase)
    sine_waves = sines * sine_amp

    # UV detection: voiced/unvoiced
    voiced_threshold = 10
    uv = (f0 > voiced_threshold).float()  # [B, L]
    # Repeat uv to match upsampled length
    uv_up = F_torch.interpolate(uv.unsqueeze(1), size=up_len, mode='nearest')  # [B, 1, up_len]

    # Apply UV mask (zero out unvoiced regions)
    sine_waves = sine_waves * uv_up

    # noise = 0 (zero noise)

    return sine_waves  # [B, H+1, up_len]


# --- Source Module ---

class SourceModule:
    def __init__(self, loader):
        self.l_linear = loader.linear("m_source.l_linear", in_d=9, out_d=1)

    @torch.no_grad()
    def forward(self, sine_waves):
        """sine_waves: [B, H+1, L] → source: [B, 1, L]"""
        # l_linear operates on the harmonic dim
        x = sine_waves.permute(0, 2, 1)  # [B, L, H+1]
        x = self.l_linear(x)  # [B, L, 1]
        x = torch.tanh(x)
        return x.permute(0, 2, 1)  # [B, 1, L]


# --- ResBlock ---

class ResBlockPT:
    def __init__(self, loader, prefix, dilations):
        self.layers = []
        for i, d in enumerate(dilations):
            alpha1, _ = loader.tensor(f"{prefix}.activations1.{i}.alpha")
            alpha2, _ = loader.tensor(f"{prefix}.activations2.{i}.alpha")
            w1, b1, _, _, _ = loader.conv1d(f"{prefix}.convs1.{i}")
            w2, b2, _, _, _ = loader.conv1d(f"{prefix}.convs2.{i}")
            self.layers.append((
                torch.from_numpy(alpha1.copy()),
                w1, b1, d,
                torch.from_numpy(alpha2.copy()),
                w2, b2,
            ))

    @torch.no_grad()
    def forward(self, x):
        for alpha1, w1, b1, d, alpha2, w2, b2 in self.layers:
            xt = snake(x, alpha1)
            xt = causal_conv1d(xt, w1, b1, 'left', dilation=d)
            xt = snake(xt, alpha2)
            xt = causal_conv1d(xt, w2, b2, 'left', dilation=1)
            x = x + xt
        return x


# --- Upsample ---

def causal_conv1d_upsample(x, weight, bias, s):
    """CausalConv1dUpsample: nearest interp by s, then right-causal conv."""
    B, C, L = x.shape
    # Nearest neighbor upsample
    x = F_torch.interpolate(x, size=L * s, mode='nearest')
    # Right-causal padding (pad right by kernel_size - 1)
    pad = weight.shape[2] - 1
    x = F_torch.pad(x, (0, pad))
    return F_torch.conv1d(x, weight, bias)


def causal_conv1d_downsample(x, weight, bias, s):
    """CausalConv1dDownSample: left-pad by (s-1), strided conv."""
    x = F_torch.pad(x, (s - 1, 0))
    return F_torch.conv1d(x, weight, bias, stride=s)


# --- Main ---

def main():
    if not os.path.exists(os.path.join(REPRO_DIR, "flow_output_final.bin")):
        print(f"ERROR: {REPRO_DIR}/flow_output_final.bin not found.")
        sys.exit(1)

    print(f"Loading GGUF: {GGUF_PATH}")
    loader = GGUFLoader(GGUF_PATH)

    # --- Load metadata ---
    nfft = loader.get_i32("istft_params.n_fft")
    hop_len = loader.get_i32("istft_params.hop_len")
    num_kernels = loader.get_i32("num_kernels")
    nb_harmonics = loader.get_i32("nb_harmonics")
    sampling_rate = loader.get_i32("sample_rate")
    nsf_alpha = loader.get_f32("nsf_alpha")
    lrelu_slope = loader.get_f32("lrelu_slope")
    audio_limit = loader.get_f32("audio_limit")
    upsample_rates = loader.get_arr("upsample_rates")
    num_upsamples = len(upsample_rates)

    scale_factor = hop_len
    for r in upsample_rates:
        scale_factor *= r

    print(f"  nfft={nfft}, hop_len={hop_len}, upsample_rates={upsample_rates}")
    print(f"  scale_factor={scale_factor}, num_kernels={num_kernels}")
    print(f"  nb_harmonics={nb_harmonics}, sampling_rate={sampling_rate}")
    print(f"  lrelu_slope={lrelu_slope}, audio_limit={audio_limit}")

    # --- Load input ---
    raw = np.fromfile(os.path.join(REPRO_DIR, "flow_output_final.bin"), dtype=np.float32)
    # ggml ne=[26, 80, 1] → numpy C-order [1, 80, 26] → PyTorch [B=1, C=80, L=26]
    n_floats = len(raw)
    mel_len = n_floats // 80
    print(f"\nInput: {n_floats} floats, mel_len={mel_len}")
    speech_feat = torch.from_numpy(raw.reshape(1, 80, mel_len).copy())
    print(f"  speech_feat: shape={list(speech_feat.shape)}, mean={speech_feat.mean():.6f}")

    # --- F0 Predictor ---
    print("\n=== F0 Predictor ===")
    f0_pred = F0Predictor(loader)
    f0 = f0_pred.forward(speech_feat)  # [1, L]
    print(f"  f0: shape={list(f0.shape)}, mean={f0.mean():.6f}, max={f0.max():.6f}")
    save_f32("hift_f0", f0[0])

    # --- NSF Source ---
    # For initial test: skip NSF and use zero source signal.
    # This isolates the decoder path (conv_pre → ups → resblocks → conv_post).
    # The C++ f0/NSF/STFT pipeline is complex and seed-dependent;
    # comparing the decoder path alone tells us if that's where the bug is.
    print("\n=== Source Signal (ZEROED for decoder-path isolation) ===")
    window = torch.hann_window(nfft)

    freq_bins = nfft // 2 + 1  # 9

    # --- Decoder ---
    print("\n=== Conv Pre ===")
    conv_pre_w, conv_pre_b, _, _, _ = loader.conv1d("conv_pre")
    x = causal_conv1d(speech_feat, conv_pre_w, conv_pre_b, 'right')
    print(f"  conv_pre out: shape={list(x.shape)}, mean={x.mean():.6f}")
    save_f32("hift_conv_pre", x[0])

    # --- Upsample loop ---
    # Load all layers
    ups_layers = []
    for i in range(num_upsamples):
        w, b, _, _, _ = loader.conv1d(f"ups.{i}")
        ups_layers.append((w, b, upsample_rates[i]))

    all_resblocks = []
    for i in range(num_upsamples * num_kernels):
        dilations = loader.get_arr(f"resblocks.{i}.dilations")
        all_resblocks.append(ResBlockPT(loader, f"resblocks.{i}", dilations))

    with torch.no_grad():
        for i in range(num_upsamples):
            print(f"\n=== Upsample Stage {i} ===")
            x = F_torch.leaky_relu(x, lrelu_slope)
            uw, ub, us = ups_layers[i]
            x = causal_conv1d_upsample(x, uw, ub, us)
            print(f"  after ups[{i}]: shape={list(x.shape)}, mean={x.mean():.6f}")

            if i == num_upsamples - 1:
                x = F_torch.pad(x, (1, 0), mode='reflect')
                print(f"  after pad_reflect: shape={list(x.shape)}")

            # Source injection: ZEROED for decoder-path isolation
            # si would be source_downs[i](s_stft) → source_resblocks[i](si)
            # With zero source, si ≈ bias-only contribution from source_downs + resblocks
            # For now we skip source entirely (equivalent to zero s_stft input)
            # TODO: implement full NSF pipeline for complete comparison

            # Resblocks average
            xs = all_resblocks[i * num_kernels].forward(x)
            for j in range(1, num_kernels):
                xs = xs + all_resblocks[i * num_kernels + j].forward(x)
            x = xs / num_kernels
            print(f"  after resblocks avg: shape={list(x.shape)}, mean={x.mean():.6f}")
            save_f32(f"hift_stage{i}", x[0])

    # --- Conv Post ---
    print("\n=== Conv Post ===")
    with torch.no_grad():
        x = F_torch.leaky_relu(x, 0.01)
        conv_post_w, conv_post_b, _, _, _ = loader.conv1d("conv_post")
        x = causal_conv1d(x, conv_post_w, conv_post_b, 'left')
    print(f"  conv_post out: shape={list(x.shape)}, mean={x.mean():.6f}")
    print(f"  conv_post min={x.min():.6f}, max={x.max():.6f}")
    save_f32("hift_conv_post", x[0])

    # --- Magnitude / Phase ---
    print("\n=== Magnitude / Phase ===")
    freq_bins = nfft // 2 + 1
    # x: [B, out_ch=nfft+2=18, L] → split into magnitude [B, 9, L] and phase [B, 9, L]
    magnitude_raw = x[:, :freq_bins, :]  # [B, 9, L]
    phase_raw = x[:, freq_bins:, :]      # [B, 9, L]

    print(f"  magnitude_raw: mean={magnitude_raw.mean():.6f}, min={magnitude_raw.min():.6f}, max={magnitude_raw.max():.6f}")
    print(f"  phase_raw: mean={phase_raw.mean():.6f}")

    magnitude = torch.exp(magnitude_raw)
    phase = torch.sin(phase_raw)

    print(f"  magnitude (exp): mean={magnitude.mean():.6f}, min={magnitude.min():.6f}, max={magnitude.max():.6f}")

    # Clamp magnitude
    magnitude = torch.clamp(magnitude, 0.0, 100.0)

    # Reconstruct complex STFT
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)

    print(f"  real: mean={real.mean():.6f}, imag: mean={imag.mean():.6f}")

    # --- iSTFT ---
    print("\n=== iSTFT ===")
    stft_complex = torch.complex(real, imag)  # [B, freq_bins, L]
    audio = torch.istft(stft_complex, n_fft=nfft, hop_length=hop_len,
                        win_length=nfft, window=window, normalized=True)
    audio = torch.clamp(audio, -audio_limit, audio_limit)
    print(f"  audio: shape={list(audio.shape)}, mean={audio.mean():.6f}")
    print(f"  audio min={audio.min():.6f}, max={audio.max():.6f}")
    save_f32("hift_audio", audio[0])

    # Save as wav for listening
    try:
        import scipy.io.wavfile
        wav_path = os.path.join(OUT, "pytorch_output.wav")
        scipy.io.wavfile.write(wav_path, sampling_rate,
                               audio[0].numpy().astype(np.float32))
        print(f"\n  Saved WAV: {wav_path}")
    except ImportError:
        print("\n  (scipy not installed, skipping WAV output)")

    print(f"\nAll gold saved to {OUT}/")


if __name__ == "__main__":
    main()
