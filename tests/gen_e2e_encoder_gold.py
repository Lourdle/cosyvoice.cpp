#!/usr/bin/env python3
"""Generate gold for encoder (build_cgraph_encode) E2E test.

Uses actual encoder input dumps from the voice clone reproduction.
Runs the encoder pipeline in PyTorch and compares against C++ encode_mu.bin.

Requires: /tmp/cv_repro3/enc_token.bin, enc_prompt_token.bin,
          enc_prompt_feat.bin, enc_embedding.bin, encode_mu.bin

Output: /tmp/cv_e2e_encoder_gold/
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
DUMP_DIR = "/tmp/cv_repro3"
OUT = "/tmp/cv_e2e_encoder_gold"
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
        return _dequant(t), list(t.shape)

    def raw_tensor(self, name):
        """Return raw data and shape without dequantization."""
        t = self._cache[name]
        return t.data, list(t.shape), t.tensor_type

    def linear(self, prefix):
        w, w_ne = self.tensor(f"{prefix}.weight")
        b, _ = self.tensor(f"{prefix}.bias")
        in_d, out_d = int(w_ne[0]), int(w_ne[1])
        layer = nn.Linear(in_d, out_d)
        layer.weight.data = torch.from_numpy(w.reshape(out_d, in_d).copy())
        layer.bias.data = torch.from_numpy(b.copy())
        return layer

    def get_metadata_int(self, key):
        for kv in self.reader.fields.values():
            if kv.name == key:
                return int(kv.parts[-1][0])
        return None


def load_dump_f32(name):
    """Load raw float32 dump (no header)."""
    path = os.path.join(DUMP_DIR, f"{name}.bin")
    return np.fromfile(path, dtype=np.float32)


def load_dump_i32(name):
    """Load I32 tensor dumped via dump_tensor_file (reads bytes as float32, reinterpret as int32)."""
    path = os.path.join(DUMP_DIR, f"{name}.bin")
    raw = np.fromfile(path, dtype=np.float32)
    return raw.view(np.int32)


def main():
    # Load encoder input dumps
    for f in ["enc_token", "enc_prompt_token", "enc_prompt_feat", "enc_embedding", "encode_mu"]:
        path = os.path.join(DUMP_DIR, f"{f}.bin")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run voice clone with COSYVOICE_DUMP_DIR first.")
            sys.exit(1)

    token_ids = load_dump_i32("enc_token")
    prompt_token_ids = load_dump_i32("enc_prompt_token")
    prompt_feat_f32 = load_dump_f32("enc_prompt_feat")
    embedding_f32 = load_dump_f32("enc_embedding")
    encode_mu_cpp = load_dump_f32("encode_mu")

    n_tokens = len(token_ids)
    n_prompt_tokens = len(prompt_token_ids)
    embed_dim = len(embedding_f32)
    prompt_mel_len = len(prompt_feat_f32) // 80

    print(f"Encoder inputs:")
    print(f"  token IDs: {n_tokens} tokens, values={token_ids[:5]}...")
    print(f"  prompt token IDs: {n_prompt_tokens} tokens, values={prompt_token_ids[:5]}...")
    print(f"  prompt_feat: {len(prompt_feat_f32)} floats ({prompt_mel_len} × 80)")
    print(f"  embedding: {embed_dim} dims")
    print(f"  encode_mu (C++ output): {len(encode_mu_cpp)} floats")

    # Load GGUF model
    print(f"\nLoading GGUF: {GGUF_PATH}")
    loader = GGUFLoader(GGUF_PATH)

    token_mel_ratio = loader.get_metadata_int("token_mel_ratio")
    pre_lookahead_len = loader.get_metadata_int("pre_lookahead_layer.pre_lookahead_len")
    print(f"token_mel_ratio={token_mel_ratio}, pre_lookahead_len={pre_lookahead_len}")

    # Load encoder weights
    spk_affine = loader.linear("spk_embed_affine_layer")

    # input_embedding: GGUF ne=[embed_dim, vocab_size] (F-order)
    ie_data, ie_shape = loader.tensor("input_embedding.weight")
    ie_embed_dim, ie_vocab_size = int(ie_shape[0]), int(ie_shape[1])
    # Reshape flat F-order data to C-order [vocab_size, embed_dim]
    input_embedding = torch.from_numpy(ie_data.reshape(ie_vocab_size, ie_embed_dim).copy())
    print(f"input_embedding: vocab={ie_vocab_size}, dim={ie_embed_dim}")

    # Pre-lookahead conv layers
    # conv1: weight [kernel_size, in_dim, out_dim] in GGUF → PyTorch Conv1d [out_ch, in_ch, kernel]
    conv1_w_data, conv1_w_shape = loader.tensor("pre_lookahead_layer.conv1.weight")
    conv1_b_data, _ = loader.tensor("pre_lookahead_layer.conv1.bias")
    k1, in1, out1 = int(conv1_w_shape[0]), int(conv1_w_shape[1]), int(conv1_w_shape[2])
    conv1 = nn.Conv1d(in1, out1, k1, bias=True, padding=0)
    conv1.weight.data = torch.from_numpy(conv1_w_data.reshape(out1, in1, k1).copy())
    conv1.bias.data = torch.from_numpy(conv1_b_data.copy())
    print(f"conv1: kernel={k1}, in={in1}, out={out1}")

    conv2_w_data, conv2_w_shape = loader.tensor("pre_lookahead_layer.conv2.weight")
    conv2_b_data, _ = loader.tensor("pre_lookahead_layer.conv2.bias")
    k2, in2, out2 = int(conv2_w_shape[0]), int(conv2_w_shape[1]), int(conv2_w_shape[2])
    conv2 = nn.Conv1d(in2, out2, k2, bias=True, padding=0)
    conv2.weight.data = torch.from_numpy(conv2_w_data.reshape(out2, in2, k2).copy())
    conv2.bias.data = torch.from_numpy(conv2_b_data.copy())
    print(f"conv2: kernel={k2}, in={in2}, out={out2}")

    # --- Run encoder in PyTorch ---
    with torch.no_grad():
        # 1. Speaker embedding: L2 norm → affine
        emb_pt = torch.from_numpy(embedding_f32.copy()).unsqueeze(0)  # [1, embed_dim]
        emb_normed = F_torch.normalize(emb_pt, p=2, dim=-1, eps=1e-6)
        spks = spk_affine(emb_normed)  # [1, out_dim]
        print(f"\nspks: shape={list(spks.shape)}, mean={spks.mean():.6f}")
        save_f32("encoder_spks", spks.squeeze(0))

        # 2. Token concat: [prompt_tokens, tokens]
        all_tokens = np.concatenate([prompt_token_ids, token_ids])
        print(f"all_tokens: {len(all_tokens)} (prompt={n_prompt_tokens} + gen={n_tokens})")

        # 3. Embedding lookup
        token_emb = input_embedding[all_tokens]  # [seq, ie_dim]
        print(f"token_emb: shape={list(token_emb.shape)}, mean={token_emb.mean():.6f}")
        save_f32("encoder_token_emb", token_emb)

        # 4. Pre-lookahead layer
        # Input: [ie_dim, seq] in ggml → PyTorch [1, ie_dim, seq]
        h = token_emb.T.unsqueeze(0)  # [1, ie_dim, seq]

        # conv1 with causal padding (pre_lookahead_len)
        h_padded = F_torch.pad(h, (pre_lookahead_len, 0))
        h_conv1 = conv1(h_padded)
        h_conv1 = F_torch.leaky_relu(h_conv1, negative_slope=0.01)

        # conv2 with causal padding (kernel_size - 1)
        h_padded2 = F_torch.pad(h_conv1, (k2 - 1, 0))
        h_conv2 = conv2(h_padded2)

        # Residual
        h_residual = h_conv2 + h
        print(f"pre_lookahead output: shape={list(h_residual.shape)}, mean={h_residual.mean():.6f}")
        save_f32("encoder_pre_lookahead", h_residual.squeeze(0).T)  # [seq, ie_dim]

        # 5. Upsample by token_mel_ratio
        # h_residual: [1, ie_dim, seq]
        seq_len = h_residual.shape[2]
        h_up = h_residual.unsqueeze(3).repeat(1, 1, 1, token_mel_ratio)  # [1, ie_dim, seq, ratio]
        h_up = h_up.reshape(1, ie_embed_dim, seq_len * token_mel_ratio)  # [1, embed_dim, total_mel_len]
        mu = h_up.squeeze(0)  # [embed_dim, total_mel_len]
        total_mel_len = mu.shape[1]
        print(f"mu (upsampled): shape={list(mu.shape)}, mean={mu.mean():.6f}")

        # 6. Conditions: pad prompt_feat to total_mel_len
        # prompt_feat from dump: ggml F-order [80, prompt_mel_len]
        prompt_feat_pt = torch.from_numpy(
            prompt_feat_f32.reshape(80, prompt_mel_len).copy()
        )  # [80, prompt_mel_len]
        mel_len2 = total_mel_len - prompt_mel_len
        conds = F_torch.pad(prompt_feat_pt, (0, mel_len2))  # [80, total_mel_len]
        print(f"conds: shape={list(conds.shape)}, mean={conds.mean():.6f}")

    # Save gold
    save_f32("encoder_mu", mu)
    save_f32("encoder_conds", conds)

    # Compare PyTorch mu against C++ encode_mu dump
    # C++ dump: ggml F-order ne=[80, total_mel_len, 1] → flat data[f + m*80]
    # PyTorch mu: C-order [80, total_mel_len] → flat data[f*total_mel_len + m]
    # Convert C++ F-order to C-order for comparison
    mu_np = mu.numpy()  # [80, total_mel_len] C-order
    mu_cpp = encode_mu_cpp.reshape(total_mel_len, ie_embed_dim).T  # → [80, total_mel_len] C-order

    print(f"\n=== Comparison: PyTorch mu vs C++ encode_mu ===")
    print(f"  PyTorch: shape={mu_np.shape}, mean={mu_np.mean():.6f}")
    print(f"  C++ (reshaped): shape={mu_cpp.shape}, mean={mu_cpp.mean():.6f}")

    diff = np.abs(mu_np - mu_cpp)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  max_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}")
    if max_diff < 0.5:
        print(f"  ✓ MATCH — encoder mu matches C++ to {max_diff:.6f} precision")
    else:
        print(f"  ✗ MISMATCH — divergence at [feat=0, mel=0..9]:")
        for m in range(min(10, total_mel_len)):
            pt_val = mu_np[0, m]
            cpp_val = mu_cpp[0, m]
            print(f"    [f=0,m={m}] PyTorch={pt_val:.6f}, C++={cpp_val:.6f}, diff={abs(pt_val-cpp_val):.6f}")

    # Also compare spks
    spks_cpp = load_dump_f32("encode_spks")
    spks_np = spks.squeeze(0).numpy()
    spks_diff = np.max(np.abs(spks_np - spks_cpp))
    print(f"\n=== Comparison: spks ===")
    print(f"  PyTorch mean={spks_np.mean():.6f}, C++ mean={spks_cpp.mean():.6f}, max_diff={spks_diff:.6f}")

    print(f"\nAll gold saved to {OUT}/")


if __name__ == "__main__":
    main()
