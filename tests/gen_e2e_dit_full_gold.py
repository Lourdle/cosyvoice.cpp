#!/usr/bin/env python3
"""Generate gold reference for FULL 22-block DiT forward pass.

Loads all 22 DiTBlock weights from GGUF, runs PyTorch forward, saves output.
This tests whether Q8_0 dequantization error accumulates across 22 blocks.

Output: /tmp/cv_e2e_dit_full_gold/
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
OUT = "/tmp/cv_e2e_dit_full_gold"
os.makedirs(OUT, exist_ok=True)

D = 1024
D_FF = 2048
HEADS = 16
HD = D // HEADS
DEPTH = 22
MEL_DIM = 80
HALF_DIM = 128
SEQ = 10  # short for tractability


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
        self._cache = {}
        for t in self.reader.tensors:
            self._cache[t.name] = t

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
    B, H, S, HD = x.shape
    freqs = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    pos = pos_ids.unsqueeze(-1).float()
    angles = pos * freqs.unsqueeze(0).unsqueeze(0)
    cos_v = torch.cos(angles).unsqueeze(1)
    sin_v = torch.sin(angles).unsqueeze(1)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * cos_v - x1 * sin_v, x0 * sin_v + x1 * cos_v], dim=-1).flatten(-2)


class PyTorchDiTBlock:
    def __init__(self, loader, block_idx):
        pfx = f"decoder.estimator.transformer_blocks.{block_idx}"
        self.adaln_linear = loader.linear(f"{pfx}.attn_norm.linear")
        self.to_q = loader.linear(f"{pfx}.attn.to_q")
        self.to_k = loader.linear(f"{pfx}.attn.to_k")
        self.to_v = loader.linear(f"{pfx}.attn.to_v")
        self.to_out = loader.linear(f"{pfx}.attn.to_out.0")
        self.ff_up = loader.linear(f"{pfx}.ff.ff.0.0")
        self.ff_down = loader.linear(f"{pfx}.ff.ff.2")

    @torch.no_grad()
    def forward(self, x, emb, pos_ids):
        B, S, D_ = x.shape
        # AdaLayerNormZero
        e = F_torch.silu(emb)
        e = self.adaln_linear(e)
        ch = e.chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa = ch[0], ch[1] + 1, ch[2]
        shift_mlp, scale_mlp, gate_mlp = ch[3], ch[4] + 1, ch[5]

        xn = F_torch.layer_norm(x, [D_], eps=1e-6)
        xn = xn * scale_msa.unsqueeze(0).unsqueeze(0) + shift_msa.unsqueeze(0).unsqueeze(0)

        # Attention
        q = self.to_q(xn).view(B, S, HEADS, HD).transpose(1, 2)
        k = self.to_k(xn).view(B, S, HEADS, HD).transpose(1, 2)
        v = self.to_v(xn).view(B, S, HEADS, HD).transpose(1, 2)
        q = apply_rope(q, pos_ids, HD)
        k = apply_rope(k, pos_ids, HD)
        a = F_torch.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).contiguous().view(B, S, D_)
        a = self.to_out(a)
        a = a * gate_msa.unsqueeze(0).unsqueeze(0)
        x = x + a

        # FeedForward
        fn = F_torch.layer_norm(x, [D_], eps=1e-6)
        fn = fn * scale_mlp.unsqueeze(0).unsqueeze(0) + shift_mlp.unsqueeze(0).unsqueeze(0)
        ff = F_torch.gelu(self.ff_up(fn))
        ff = self.ff_down(ff)
        ff = ff * gate_mlp.unsqueeze(0).unsqueeze(0)
        x = x + ff
        return x


def main():
    print(f"Loading GGUF: {GGUF_PATH}")
    loader = GGUFLoader(GGUF_PATH)

    # Load all 22 blocks
    print(f"Loading {DEPTH} DiTBlocks...")
    blocks = []
    for i in range(DEPTH):
        blocks.append(PyTorchDiTBlock(loader, i))
        if (i + 1) % 5 == 0:
            print(f"  loaded {i+1}/{DEPTH}")

    # Load norm_out + proj_out
    norm_out_linear = loader.linear("decoder.estimator.norm_out.linear")
    proj_out = loader.linear("decoder.estimator.proj_out")

    # TimestepEmbedding
    time_mlp_0 = loader.linear("decoder.estimator.time_embed.time_mlp.0")
    time_mlp_2 = loader.linear("decoder.estimator.time_embed.time_mlp.2")

    emb_coeff = torch.exp(
        torch.arange(HALF_DIM, dtype=torch.float32) * (-math.log(10000) / (HALF_DIM - 1))
    ) * 1000.0

    print("All weights loaded.\n")

    # --- Generate timestep embedding ---
    t_val = 0.019
    sinpos = torch.cat([
        torch.sin(torch.tensor([t_val]) * emb_coeff),
        torch.cos(torch.tensor([t_val]) * emb_coeff),
    ])
    with torch.no_grad():
        te = F_torch.silu(time_mlp_0(sinpos))
        te = time_mlp_2(te)  # [D]

    # --- Input (random, deterministic) ---
    torch.manual_seed(46)
    # After InputEmbedding, x would be [B=1, SEQ, D=1024]
    x = torch.randn(1, SEQ, D)
    pos_ids = torch.arange(SEQ).unsqueeze(0)

    save_f32("full_dit_input", x[0])
    save_f32("full_dit_time_emb", te)

    # --- Forward through all 22 blocks ---
    print("Running 22-block DiT forward...")
    with torch.no_grad():
        for i, block in enumerate(blocks):
            x = block.forward(x, te, pos_ids)
            if i in (0, 5, 10, 15, 21):
                m = x.mean().item()
                s = x.std().item()
                mx = x.abs().max().item()
                print(f"  after block {i:2d}: mean={m:+.6f}, std={s:.6f}, max_abs={mx:.6f}")
                save_f32(f"full_dit_after_block{i:02d}", x[0])

    # --- AdaLayerNorm_Final ---
    with torch.no_grad():
        e_final = F_torch.silu(te)
        e_final = norm_out_linear(e_final)
        scale, shift = e_final.chunk(2, dim=-1)
        scale = scale + 1
        x = F_torch.layer_norm(x, [D], eps=1e-6)
        x = x * scale.unsqueeze(0).unsqueeze(0) + shift.unsqueeze(0).unsqueeze(0)

    # --- proj_out ---
    with torch.no_grad():
        x = proj_out(x)  # [1, SEQ, 80]

    save_f32("full_dit_output", x[0])
    print(f"\n  final output: shape={list(x.shape)}, mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"\nAll gold saved to {OUT}/")


if __name__ == "__main__":
    main()
