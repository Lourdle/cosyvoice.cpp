#!/usr/bin/env python3
"""Verify conditioned vs unconditioned velocity through DiT using GGUF weights.

This script loads the GGUF model into PyTorch and runs the DiT forward pass
with realistic (non-zero) conditioning inputs to determine whether v_cond=-9
is a property of the weights or a C++ implementation bug.

If v_cond ≈ -9 here too: the weights/model produce this — not a C++ bug.
If v_cond ≈ -5 here: the C++ implementation has a bug in the conditioning path.
"""
import math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import gguf

GGUF_PATH = os.environ.get(
    "COSYVOICE_TEST_MODEL",
    os.path.join(os.path.dirname(__file__), "../models/tts/cosyvoice/CosyVoice3-2512_Q8_0.gguf"),
)

D = 1024
HEADS = 16
HD = D // HEADS
DEPTH = 22
MEL_DIM = 80
HALF_DIM = 128
SEQ_LEN = 40  # enough frames for meaningful test

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
        return _dequant(t), list(t.shape)

    def linear(self, prefix):
        w, w_ne = self.tensor(f"{prefix}.weight")
        b, _ = self.tensor(f"{prefix}.bias")
        in_d, out_d = int(w_ne[0]), int(w_ne[1])
        layer = nn.Linear(in_d, out_d)
        layer.weight.data = torch.from_numpy(w.reshape(out_d, in_d).copy())
        layer.bias.data = torch.from_numpy(b.copy())
        return layer

def apply_rope(x, pos_ids, hd):
    B, H, S, HD_ = x.shape
    freqs = 1.0 / (10000.0 ** (torch.arange(0, HD_, 2, dtype=torch.float32) / HD_))
    pos = pos_ids.unsqueeze(-1).float()
    angles = pos * freqs.unsqueeze(0).unsqueeze(0)
    cos_v = torch.cos(angles).unsqueeze(1)
    sin_v = torch.sin(angles).unsqueeze(1)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * cos_v - x1 * sin_v, x0 * sin_v + x1 * cos_v], dim=-1).flatten(-2)

class DiTBlock:
    def __init__(self, loader, idx):
        pfx = f"decoder.estimator.transformer_blocks.{idx}"
        self.adaln = loader.linear(f"{pfx}.attn_norm.linear")
        self.to_q = loader.linear(f"{pfx}.attn.to_q")
        self.to_k = loader.linear(f"{pfx}.attn.to_k")
        self.to_v = loader.linear(f"{pfx}.attn.to_v")
        self.to_out = loader.linear(f"{pfx}.attn.to_out.0")
        self.ff_up = loader.linear(f"{pfx}.ff.ff.0.0")
        self.ff_down = loader.linear(f"{pfx}.ff.ff.2")

    @torch.no_grad()
    def forward(self, x, emb, pos_ids):
        B, S, D_ = x.shape
        e = self.adaln(F_torch.silu(emb))
        ch = e.chunk(6, dim=-1)
        sh_m, sc_m, g_m = ch[0], ch[1]+1, ch[2]
        sh_f, sc_f, g_f = ch[3], ch[4]+1, ch[5]

        xn = F_torch.layer_norm(x, [D_], eps=1e-6)
        xn = xn * sc_m.unsqueeze(0).unsqueeze(0) + sh_m.unsqueeze(0).unsqueeze(0)

        q = self.to_q(xn).view(B, S, HEADS, HD).transpose(1, 2)
        k = self.to_k(xn).view(B, S, HEADS, HD).transpose(1, 2)
        v = self.to_v(xn).view(B, S, HEADS, HD).transpose(1, 2)
        q = apply_rope(q, pos_ids, HD)
        k = apply_rope(k, pos_ids, HD)
        a = F_torch.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).contiguous().view(B, S, D_)
        a = self.to_out(a) * g_m.unsqueeze(0).unsqueeze(0)
        x = x + a

        fn = F_torch.layer_norm(x, [D_], eps=1e-6)
        fn = fn * sc_f.unsqueeze(0).unsqueeze(0) + sh_f.unsqueeze(0).unsqueeze(0)
        ff = self.ff_down(F_torch.gelu(self.ff_up(fn)))
        x = x + ff * g_f.unsqueeze(0).unsqueeze(0)
        return x


def main():
    print(f"Loading GGUF: {GGUF_PATH}")
    loader = GGUFLoader(GGUF_PATH)

    print(f"Loading {DEPTH} DiTBlocks...")
    blocks = [DiTBlock(loader, i) for i in range(DEPTH)]
    norm_out_linear = loader.linear("decoder.estimator.norm_out.linear")
    proj_out = loader.linear("decoder.estimator.proj_out")
    ie_proj = loader.linear("decoder.estimator.input_embed.proj")

    time_mlp_0 = loader.linear("decoder.estimator.time_embed.time_mlp.0")
    time_mlp_2 = loader.linear("decoder.estimator.time_embed.time_mlp.2")
    emb_coeff = torch.exp(
        torch.arange(HALF_DIM, dtype=torch.float32) * (-math.log(10000) / (HALF_DIM - 1))
    ) * 1000.0

    # Read cfg_rate from metadata
    cfg_rate = 0.7
    for kv in loader.reader.fields.values():
        if kv.name == "decoder.inference_cfg_rate":
            cfg_rate = float(kv.parts[-1][0])
    print(f"cfg_rate={cfg_rate}")

    # t_span: cosine schedule (same as C++)
    t_span = [1.0 - math.cos(0.1 * 0.5 * math.pi * i) for i in range(11)]

    # --- Create realistic inputs ---
    torch.manual_seed(42)

    # x: random noise at t=0 (start of flow matching)
    x = torch.randn(1, SEQ_LEN, MEL_DIM)

    # cond: mel spectrogram values (typical range -5 to -15, mean ≈ -5)
    cond = torch.randn(1, SEQ_LEN, MEL_DIM) * 3.0 - 5.0

    # mu: encoder hidden state (typically small values around 0)
    mu = torch.randn(1, SEQ_LEN, MEL_DIM) * 0.5

    # spks: projected speaker embedding (small, mean ≈ 0)
    spks = torch.randn(1, 1, MEL_DIM) * 0.1

    print(f"\n--- Inputs ---")
    print(f"  x:    mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"  cond: mean={cond.mean():.4f}, std={cond.std():.4f}")
    print(f"  mu:   mean={mu.mean():.4f}, std={mu.std():.4f}")
    print(f"  spks: mean={spks.mean():.4f}, std={spks.std():.4f}")

    # --- Run DiT for multiple ODE steps ---
    for step in range(1, 4):  # Steps 1, 2, 3
        t = t_span[step - 1]
        dt = t_span[step] - t_span[step - 1]

        # Timestep embedding
        with torch.no_grad():
            sinpos = torch.cat([
                torch.sin(torch.tensor([t]) * emb_coeff),
                torch.cos(torch.tensor([t]) * emb_coeff),
            ])
            te = time_mlp_2(F_torch.silu(time_mlp_0(sinpos)))

        # Batch=2 for CFG
        x_b2 = x.repeat(2, 1, 1)

        # batch 0 = conditioned (real values), batch 1 = unconditioned (zeros)
        mu_b2 = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_b2 = torch.cat([spks.expand(1, SEQ_LEN, MEL_DIM), torch.zeros(1, SEQ_LEN, MEL_DIM)], dim=0)
        cond_b2 = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        # InputEmbedding
        ie_input = torch.cat([x_b2, cond_b2, mu_b2, spks_b2], dim=2)  # [2, SEQ, 320]

        with torch.no_grad():
            ie_out = ie_proj(ie_input)
            # NOTE: skipping conv_pos_embed for simplicity

            pos_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(2, -1)
            h = ie_out
            for block in blocks:
                h = block.forward(h, te, pos_ids)

            # AdaLayerNorm_Final
            e_f = norm_out_linear(F_torch.silu(te))
            scale_f, shift_f = e_f.chunk(2, dim=-1)
            scale_f = scale_f + 1
            h = F_torch.layer_norm(h, [D], eps=1e-6)
            h = h * scale_f.unsqueeze(0).unsqueeze(0) + shift_f.unsqueeze(0).unsqueeze(0)

            # proj_out
            dit_out = proj_out(h)  # [2, SEQ, 80]

        v_cond = dit_out[0:1]
        v_uncond = dit_out[1:2]
        v_cfg = (1 + cfg_rate) * v_cond - cfg_rate * v_uncond

        print(f"\n--- ODE Step {step} (t={t:.6f}, dt={dt:.6f}) ---")
        print(f"  v_cond:   mean={v_cond.mean():.4f}, std={v_cond.std():.4f}")
        print(f"  v_uncond: mean={v_uncond.mean():.4f}, std={v_uncond.std():.4f}")
        print(f"  v_cfg:    mean={v_cfg.mean():.4f}")
        print(f"  x before: mean={x.mean():.4f}")

        # Euler step
        x = x + dt * v_cfg
        print(f"  x after:  mean={x.mean():.4f}")

    print(f"\n--- Final Result ---")
    print(f"  flow_output: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"  Expected: mean ≈ -5 (good mel spectrogram range)")
    print(f"  If mean ≈ -10: weights/model are the cause, not C++ impl")


if __name__ == "__main__":
    main()
