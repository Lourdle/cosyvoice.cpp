#!/usr/bin/env python3
"""Generate gold for a single ODE step: full DiT (22 blocks, batch=2 CFG).

Uses the actual encode_mu/step1_feat dumps from the voice clone reproduction
as inputs, runs through the same DiT + CFG logic in PyTorch, and saves the
expected step2 output. This tests whether the ODE drift appears in a single
step or only accumulates over multiple iterations.

Requires: /tmp/cv_repro3/encode_mu.bin and step1_feat.bin from the voice clone
reproduction run.

Output: /tmp/cv_e2e_ode_gold/
"""
import math, os, struct, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import gguf

# Reuse the loader and block classes from gen_e2e_dit_full_gold.py
GGUF_PATH = os.environ.get(
    "COSYVOICE_TEST_MODEL",
    os.path.join(os.path.dirname(__file__), "../../models/tts/cosyvoice/CosyVoice3-2512_Q8_0.gguf"),
)
OUT = "/tmp/cv_e2e_ode_gold"
os.makedirs(OUT, exist_ok=True)

D = 1024
HEADS = 16
HD = D // HEADS
DEPTH = 22
MEL_DIM = 80
HALF_DIM = 128


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
    # Load dumps from voice clone reproduction
    repro_dir = "/tmp/cv_repro3"
    if not os.path.exists(os.path.join(repro_dir, "step1_feat.bin")):
        print(f"ERROR: {repro_dir}/step1_feat.bin not found. Run voice clone first.")
        sys.exit(1)

    step1_data = np.fromfile(os.path.join(repro_dir, "step1_feat.bin"), dtype=np.float32)
    encode_mu = np.fromfile(os.path.join(repro_dir, "encode_mu.bin"), dtype=np.float32)

    # step1_feat shape: [seq, 80, 1] from dump (ggml ne order)
    # encode_mu shape: [80, seq, 1] from dump
    n_floats_step1 = len(step1_data)
    seq_len = n_floats_step1 // 80
    print(f"step1: {n_floats_step1} floats, seq_len={seq_len}")
    print(f"encode_mu: {len(encode_mu)} floats")

    # Use a small subset for tractability
    SUB_SEQ = 20  # first 20 time steps only
    print(f"Using first {SUB_SEQ} time steps for tractability")

    print(f"\nLoading GGUF: {GGUF_PATH}")
    loader = GGUFLoader(GGUF_PATH)

    # Load all blocks
    print(f"Loading {DEPTH} DiTBlocks...")
    blocks = [DiTBlock(loader, i) for i in range(DEPTH)]

    norm_out_linear = loader.linear("decoder.estimator.norm_out.linear")
    proj_out = loader.linear("decoder.estimator.proj_out")

    # InputEmbedding components
    ie_proj = loader.linear("decoder.estimator.input_embed.proj")
    # conv_pos_embed loaded separately (grouped conv is complex to replicate)
    # For this test, we skip conv_pos_embed and test the DiT portion only

    # TimestepEmbedding
    time_mlp_0 = loader.linear("decoder.estimator.time_embed.time_mlp.0")
    time_mlp_2 = loader.linear("decoder.estimator.time_embed.time_mlp.2")
    emb_coeff = torch.exp(
        torch.arange(HALF_DIM, dtype=torch.float32) * (-math.log(10000) / (HALF_DIM - 1))
    ) * 1000.0

    # Read cfg rate
    cfg_rate = 0.7
    for kv in loader.reader.fields.values():
        if kv.name == "decoder.inference_cfg_rate":
            cfg_rate = float(kv.parts[-1][0])
    print(f"cfg_rate={cfg_rate}")

    # t_span: cosine schedule
    t_span = [1.0 - math.cos(0.1 * 0.5 * math.pi * i) for i in range(11)]
    print(f"t_span[0:3]={t_span[0:3]}")

    # --- Simulate ODE step 2 ---
    # step 2 uses t=t_span[0]+dt_1 = t_span[1], dt=t_span[2]-t_span[1]
    t_step2 = t_span[1]  # accumulated t after step 1
    dt_step2 = t_span[2] - t_span[1]
    print(f"step2: t={t_step2:.6f}, dt={dt_step2:.6f}")

    # Timestep embedding for step 2
    with torch.no_grad():
        sinpos = torch.cat([
            torch.sin(torch.tensor([t_step2]) * emb_coeff),
            torch.cos(torch.tensor([t_step2]) * emb_coeff),
        ])
        te = time_mlp_2(F_torch.silu(time_mlp_0(sinpos)))  # [D=1024]
    save_f32("ode_step2_time_emb", te)

    # x = step1_feat [seq, 80] in ggml dump format
    # ggml ne=[seq, 80]: F-order data[s + f*seq]
    # To convert to PyTorch [B, seq, 80]: need to reshape
    # Dump flat bytes = ggml F-order ne=[seq, 80]: data[s + f*seq]
    # = numpy C-order [80, seq]: data[f*seq + s]
    # So reshape as [80, seq] then transpose to [seq, 80]
    x_np = step1_data.reshape(80, seq_len).T[:SUB_SEQ, :]  # [SUB_SEQ, 80]
    x_pt = torch.from_numpy(x_np.copy()).unsqueeze(0)  # [1, SUB_SEQ, 80]

    # mu = encode_mu [80, seq, 1] in ggml dump format
    mu_np = encode_mu.reshape(80, seq_len).T[:SUB_SEQ, :]  # [SUB_SEQ, 80]
    mu_pt = torch.from_numpy(mu_np.copy()).unsqueeze(0)  # [1, SUB_SEQ, 80]

    # For simplicity, use zero spks (the actual spks were dumped but format is [80,1,1])
    spks_pt = torch.zeros(1, 1, 80)  # broadcast to all time steps

    # cond = same structure as mu but from conds dump (not saved, use zeros for now)
    cond_pt = torch.zeros(1, SUB_SEQ, 80)

    print(f"\nInputs: x=[{x_pt.shape}], mu=[{mu_pt.shape}]")
    print(f"  x mean={x_pt.mean():.6f}, mu mean={mu_pt.mean():.6f}")

    # Save mu as gold (so C++ test doesn't need raw dumps)
    save_f32("ode_step2_mu", mu_pt[0])

    # --- InputEmbedding (simplified, no conv_pos_embed for this test) ---
    # In real pipeline: concat(x, cond, mu, spks) → proj → conv_pos_embed + residual
    # Total input dim = 80 + 80 + 80 + 80 = 320 (matches proj input dim)

    # Batch=2 for CFG: repeat x, zero out conditioning for batch 1
    x_b2 = x_pt.repeat(2, 1, 1)  # [2, SUB_SEQ, 80]

    # Conditioning: batch 0 = real, batch 1 = zeros
    mu_b2 = torch.cat([mu_pt, torch.zeros_like(mu_pt)], dim=0)  # [2, SUB_SEQ, 80]
    spks_b2 = torch.cat([spks_pt.expand(1, SUB_SEQ, 80), torch.zeros(1, SUB_SEQ, 80)], dim=0)
    cond_b2 = torch.cat([cond_pt, torch.zeros_like(cond_pt)], dim=0)

    # Concat along feature dim: [2, SUB_SEQ, 320]
    ie_input = torch.cat([x_b2, cond_b2, mu_b2, spks_b2], dim=2)

    with torch.no_grad():
        # proj: [2, SUB_SEQ, 320] → [2, SUB_SEQ, 1024]
        ie_out = ie_proj(ie_input)
        # Skip conv_pos_embed for this test (already verified separately)

    print(f"  IE output: mean={ie_out.mean():.6f}, std={ie_out.std():.6f}")
    print(f"  IE batch0 mean={ie_out[0].mean():.6f}, batch1 mean={ie_out[1].mean():.6f}")
    save_f32("ode_step2_ie_output", ie_out)  # [2, SUB_SEQ, 1024]

    # --- DiT forward with batch=2 ---
    pos_ids = torch.arange(SUB_SEQ).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        h = ie_out
        for i, block in enumerate(blocks):
            h = block.forward(h, te, pos_ids)

        # AdaLayerNorm_Final
        e_f = norm_out_linear(F_torch.silu(te))
        scale_f, shift_f = e_f.chunk(2, dim=-1)
        scale_f = scale_f + 1
        h = F_torch.layer_norm(h, [D], eps=1e-6)
        h = h * scale_f.unsqueeze(0).unsqueeze(0) + shift_f.unsqueeze(0).unsqueeze(0)

        # proj_out
        dit_out = proj_out(h)  # [2, SUB_SEQ, 80]

    # --- CFG ---
    dphi_cond = dit_out[0:1]   # [1, SUB_SEQ, 80]
    dphi_uncond = dit_out[1:2]

    dphi_final = (1 + cfg_rate) * dphi_cond - cfg_rate * dphi_uncond

    # Euler step
    x_new = x_pt + dt_step2 * dphi_final

    print(f"\n=== ODE Step 2 Results ===")
    print(f"  dphi_cond mean={dphi_cond.mean():.6f}, std={dphi_cond.std():.6f}")
    print(f"  dphi_uncond mean={dphi_uncond.mean():.6f}, std={dphi_uncond.std():.6f}")
    print(f"  dphi_final mean={dphi_final.mean():.6f}")
    print(f"  x_new mean={x_new.mean():.6f} (x_old mean={x_pt.mean():.6f})")
    print(f"  mean_shift = {(x_new.mean() - x_pt.mean()):.6f}")

    save_f32("ode_step2_x_old", x_pt[0])
    save_f32("ode_step2_dphi_cond", dphi_cond[0])
    save_f32("ode_step2_dphi_uncond", dphi_uncond[0])
    save_f32("ode_step2_dphi_final", dphi_final[0])
    save_f32("ode_step2_x_new", x_new[0])

    print(f"\nAll gold saved to {OUT}/")


if __name__ == "__main__":
    main()
