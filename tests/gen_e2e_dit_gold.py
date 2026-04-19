#!/usr/bin/env python3
"""Generate E2E gold references for DiT pipeline modules using REAL GGUF weights.

Levels:
  1. TimestepEmbedding: SinusPosEmb → MLP
  2. Attention (block 0): QKV → RoPE → SDPA → Out
  3. FeedForward (block 0): Linear → GELU → Linear
  4. AdaLayerNormZero (block 0): SiLU → Linear → split → LN*scale+shift
  5. DiTBlock (block 0): AdaLN → Attn → FF → residual

Output: binary f32 files under /tmp/cv_e2e_dit_gold/
Format: [ndim: u32] [shape[0]: i64] ... [shape[n-1]: i64] [flat f32 data]
"""
import math, os, struct, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import gguf

# --- Config ---
GGUF_PATH = os.environ.get(
    "COSYVOICE_TEST_MODEL",
    os.path.join(os.path.dirname(__file__), "../../models/tts/cosyvoice/CosyVoice3-2512_Q8_0.gguf"),
)
OUT = "/tmp/cv_e2e_dit_gold"
os.makedirs(OUT, exist_ok=True)

D = 1024       # d_model
D_FF = 2048    # feedforward dim
HEADS = 16
HEAD_DIM = D // HEADS  # 64
SEQ = 10       # short sequence for testing
HALF_DIM = 128 # SinusPosEmb half_dim


def save_f32(name, data):
    """Save numpy array as headerful binary f32."""
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


def load_tensor(reader, name):
    """Load GGUF tensor, dequantize to F32, return flat numpy."""
    for t in reader.tensors:
        if t.name == name:
            ttype = t.tensor_type
            if ttype in (0, 1):
                return t.data.astype(np.float32).flatten(), list(t.shape)
            else:
                return gguf.dequantize(t.data, gguf.GGMLQuantizationType(ttype)).flatten(), list(t.shape)
    raise KeyError(f"Tensor {name!r} not found")


def load_linear(reader, prefix):
    """Load a Linear layer from GGUF. Returns (weight_pt, bias_pt, in_dim, out_dim)."""
    w, w_ne = load_tensor(reader, f"{prefix}.weight")
    b, _ = load_tensor(reader, f"{prefix}.bias")
    in_dim, out_dim = int(w_ne[0]), int(w_ne[1])
    # GGUF F-order ne=[in, out] = PyTorch C-order [out, in]
    w_pt = torch.from_numpy(w.reshape(out_dim, in_dim).copy())
    b_pt = torch.from_numpy(b.copy())
    return w_pt, b_pt, in_dim, out_dim


def make_linear(reader, prefix):
    """Create a PyTorch Linear layer with GGUF weights."""
    w_pt, b_pt, in_dim, out_dim = load_linear(reader, prefix)
    layer = nn.Linear(in_dim, out_dim)
    layer.weight.data = w_pt
    layer.bias.data = b_pt
    return layer


def apply_rope(x, position_ids, head_dim):
    """Apply standard RoPE (matches ggml GGML_ROPE_TYPE_NORMAL).

    x: [B, heads, seq, head_dim]
    position_ids: [B, seq] (int)
    """
    B, H, S, HD = x.shape
    freqs = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    # position_ids: [B, S] -> [B, S, 1]
    pos = position_ids.unsqueeze(-1).float()  # [B, S, 1]
    # angles: [B, S, HD/2]
    angles = pos * freqs.unsqueeze(0).unsqueeze(0)  # [B, S, HD//2]
    cos_vals = torch.cos(angles)  # [B, S, HD//2]
    sin_vals = torch.sin(angles)  # [B, S, HD//2]
    # Reshape for broadcast: [B, 1, S, HD//2]
    cos_vals = cos_vals.unsqueeze(1)
    sin_vals = sin_vals.unsqueeze(1)
    # x pairs: x[..., 0::2], x[..., 1::2]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    # Rotate
    y0 = x0 * cos_vals - x1 * sin_vals
    y1 = x0 * sin_vals + x1 * cos_vals
    # Interleave back
    out = torch.stack([y0, y1], dim=-1).flatten(-2)
    return out


def main():
    print(f"Loading GGUF: {GGUF_PATH}")
    reader = gguf.GGUFReader(GGUF_PATH)

    # Read metadata
    for kv in reader.fields.values():
        if kv.name == "decoder.estimator.heads":
            heads_meta = int(kv.parts[-1][0])
            assert heads_meta == HEADS, f"Expected {HEADS} heads, got {heads_meta}"

    torch.manual_seed(44)

    # ===========================================
    # Level 1: TimestepEmbedding
    # GIVEN: GGUF time_mlp weights + t=0.019
    # WHEN:  SinusPosEmb(t) → Linear → SiLU → Linear
    # THEN:  output matches PyTorch
    # ===========================================
    print("\n--- Level 1: TimestepEmbedding ---")

    time_mlp_0 = make_linear(reader, "decoder.estimator.time_embed.time_mlp.0")
    time_mlp_2 = make_linear(reader, "decoder.estimator.time_embed.time_mlp.2")

    # SinusPosEmb: computed, not stored in GGUF
    # half_dim=128, emb[i] = exp(i * -log(10000) / (half_dim-1)) * 1000
    emb_coeff = torch.exp(torch.arange(HALF_DIM, dtype=torch.float32) * (-math.log(10000) / (HALF_DIM - 1))) * 1000.0

    t_val = 0.019  # second point of cosine t_span
    t_tensor = torch.tensor([t_val], dtype=torch.float32)

    # SinusPosEmb forward: [sin(t*emb), cos(t*emb)]
    sinpos = torch.cat([
        torch.sin(t_tensor * emb_coeff),
        torch.cos(t_tensor * emb_coeff),
    ])  # [256]

    # MLP: Linear → SiLU → Linear
    te_h = time_mlp_0(sinpos)
    te_h = F_torch.silu(te_h)
    te_out = time_mlp_2(te_h)

    save_f32("level1_t_input", t_tensor.numpy())
    save_f32("level1_sinpos", sinpos.numpy())
    save_f32("level1_time_emb", te_out)  # [1024]
    save_f32("level1_emb_coeff", emb_coeff)  # [128] for C++ to verify

    print(f"  sinpos: mean={sinpos.mean():.6f}, std={sinpos.std():.6f}")
    print(f"  time_emb: mean={te_out.mean():.6f}, std={te_out.std():.6f}")

    # ===========================================
    # Level 2: Attention (block 0)
    # GIVEN: GGUF Q/K/V/Out weights + input [D, SEQ, batch]
    # WHEN:  QKV → RoPE → SDPA → Out
    # THEN:  output matches PyTorch
    # ===========================================
    print("\n--- Level 2: Attention (block 0) ---")

    prefix = "decoder.estimator.transformer_blocks.0.attn"
    to_q = make_linear(reader, f"{prefix}.to_q")
    to_k = make_linear(reader, f"{prefix}.to_k")
    to_v = make_linear(reader, f"{prefix}.to_v")
    to_out = make_linear(reader, f"{prefix}.to_out.0")

    # Input: [B=1, SEQ, D] for PyTorch
    attn_input = torch.randn(1, SEQ, D)

    # QKV projection
    q = to_q(attn_input)  # [B, SEQ, D]
    k = to_k(attn_input)
    v = to_v(attn_input)

    # Reshape for multi-head: [B, SEQ, heads, head_dim] → [B, heads, SEQ, head_dim]
    q = q.view(1, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(1, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(1, SEQ, HEADS, HEAD_DIM).transpose(1, 2)

    # Position IDs
    position_ids = torch.arange(SEQ).unsqueeze(0)  # [1, SEQ]

    # Apply RoPE
    q_rope = apply_rope(q, position_ids, HEAD_DIM)
    k_rope = apply_rope(k, position_ids, HEAD_DIM)

    # Scaled dot product attention
    attn_out = F_torch.scaled_dot_product_attention(q_rope, k_rope, v)  # [B, heads, SEQ, head_dim]

    # Reshape back: [B, SEQ, D]
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, SEQ, D)

    # Output projection
    attn_final = to_out(attn_out)  # [B, SEQ, D]

    # Save in ggml layout:
    # ggml input ne=[D, SEQ, B]: PyTorch [B, SEQ, D] → save [B, SEQ, D] numpy C-order
    # which = ggml [D, SEQ, B] F-order bytes (same flat index)
    save_f32("level2_attn_input", attn_input[0])  # [SEQ, D]
    save_f32("level2_attn_q_rope", q_rope[0])     # [heads, SEQ, head_dim]
    save_f32("level2_attn_output", attn_final[0])  # [SEQ, D]

    print(f"  q_rope: mean={q_rope.mean():.6f}, std={q_rope.std():.6f}")
    print(f"  attn_output: mean={attn_final.mean():.6f}, std={attn_final.std():.6f}")

    # ===========================================
    # Level 3: FeedForward (block 0)
    # GIVEN: GGUF FF weights + input [D, SEQ, 1]
    # WHEN:  Linear → GELU → Linear
    # THEN:  output matches PyTorch
    # ===========================================
    print("\n--- Level 3: FeedForward (block 0) ---")

    ff_up = make_linear(reader, "decoder.estimator.transformer_blocks.0.ff.ff.0.0")
    ff_down = make_linear(reader, "decoder.estimator.transformer_blocks.0.ff.ff.2")

    ff_input = torch.randn(1, SEQ, D)
    ff_h = ff_up(ff_input)
    ff_h = F_torch.gelu(ff_h)
    ff_out = ff_down(ff_h)

    save_f32("level3_ff_input", ff_input[0])  # [SEQ, D]
    save_f32("level3_ff_output", ff_out[0])    # [SEQ, D]

    print(f"  ff_output: mean={ff_out.mean():.6f}, std={ff_out.std():.6f}")

    # ===========================================
    # Level 4: AdaLayerNormZero (block 0)
    # GIVEN: GGUF attn_norm weights + x [D, SEQ, 1] + emb [D]
    # WHEN:  SiLU(emb) → Linear → split×6 → LN(x)*scale+shift
    # THEN:  norm output + gates match PyTorch
    # ===========================================
    print("\n--- Level 4: AdaLayerNormZero (block 0) ---")

    adaln_linear = make_linear(reader, "decoder.estimator.transformer_blocks.0.attn_norm.linear")

    adaln_x = torch.randn(1, SEQ, D)
    adaln_emb = te_out.clone().detach()  # use Level 1 output as time embedding

    # SiLU → Linear → split×6
    emb_silu = F_torch.silu(adaln_emb)
    emb_proj = adaln_linear(emb_silu)  # [6*D = 6144]
    chunks = emb_proj.chunk(6, dim=-1)  # 6 × [D]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

    # scale + 1
    scale_msa = scale_msa + 1.0
    scale_mlp_out = scale_mlp + 1.0

    # LayerNorm (no affine) + scale + shift
    x_norm = F_torch.layer_norm(adaln_x, [D], eps=1e-6)
    # scale_msa [D], x_norm [B, SEQ, D] → unsqueeze to [1, 1, D] for broadcast
    norm_out = x_norm * scale_msa.unsqueeze(0).unsqueeze(0) + shift_msa.unsqueeze(0).unsqueeze(0)

    save_f32("level4_adaln_x", adaln_x[0])     # [SEQ, D]
    save_f32("level4_adaln_emb", adaln_emb)     # [D]
    save_f32("level4_adaln_norm_out", norm_out[0]) # [SEQ, D]
    save_f32("level4_adaln_gate_msa", gate_msa)    # [D]
    save_f32("level4_adaln_shift_mlp", shift_mlp)  # [D]
    save_f32("level4_adaln_scale_mlp", scale_mlp_out)  # [D] (after +1)
    save_f32("level4_adaln_gate_mlp", gate_mlp)    # [D]

    print(f"  norm_out: mean={norm_out.mean():.6f}, std={norm_out.std():.6f}")
    print(f"  gate_msa: mean={gate_msa.mean():.6f}")

    # ===========================================
    # Level 5: Full DiTBlock (block 0)
    # GIVEN: All block 0 weights + x [D, SEQ, 1] + time_emb [D] + position_ids
    # WHEN:  AdaLN → Attn → FF → residual
    # THEN:  output matches PyTorch
    # ===========================================
    print("\n--- Level 5: DiTBlock (block 0) ---")

    # Load ff_norm (second LayerNorm in block)
    # From the GGUF exploration: no separate norm tensor → elementwise_affine=False

    dit_x = torch.randn(1, SEQ, D)
    dit_emb = te_out.clone().detach()
    dit_pos = torch.arange(SEQ).unsqueeze(0)  # [1, SEQ]

    # --- AdaLayerNormZero ---
    emb_s = F_torch.silu(dit_emb)
    emb_p = adaln_linear(emb_s)
    ch = emb_p.chunk(6, dim=-1)
    d_shift_msa, d_scale_msa, d_gate_msa = ch[0], ch[1] + 1.0, ch[2]
    d_shift_mlp, d_scale_mlp, d_gate_mlp = ch[3], ch[4] + 1.0, ch[5]

    x_n = F_torch.layer_norm(dit_x, [D], eps=1e-6)
    attn_input_norm = x_n * d_scale_msa.unsqueeze(0).unsqueeze(0) + d_shift_msa.unsqueeze(0).unsqueeze(0)

    # --- Attention ---
    q5 = to_q(attn_input_norm)
    k5 = to_k(attn_input_norm)
    v5 = to_v(attn_input_norm)
    q5 = q5.view(1, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    k5 = k5.view(1, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    v5 = v5.view(1, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    q5 = apply_rope(q5, dit_pos, HEAD_DIM)
    k5 = apply_rope(k5, dit_pos, HEAD_DIM)
    a5 = F_torch.scaled_dot_product_attention(q5, k5, v5)
    a5 = a5.transpose(1, 2).contiguous().view(1, SEQ, D)
    attn5 = to_out(a5)

    # Gate + residual
    attn5 = attn5 * d_gate_msa.unsqueeze(0).unsqueeze(0)
    dit_x2 = dit_x + attn5

    # --- FeedForward ---
    ff_n = F_torch.layer_norm(dit_x2, [D], eps=1e-6)
    ff_n = ff_n * d_scale_mlp.unsqueeze(0).unsqueeze(0) + d_shift_mlp.unsqueeze(0).unsqueeze(0)
    ff5_h = F_torch.gelu(ff_up(ff_n))
    ff5_out = ff_down(ff5_h)
    ff5_out = ff5_out * d_gate_mlp.unsqueeze(0).unsqueeze(0)
    dit_out = dit_x2 + ff5_out

    save_f32("level5_dit_x", dit_x[0])        # [SEQ, D]
    save_f32("level5_dit_emb", dit_emb)        # [D]
    save_f32("level5_dit_output", dit_out[0])    # [SEQ, D]
    save_f32("level5_dit_attn_out", attn5[0])    # [SEQ, D]

    print(f"  dit_output: mean={dit_out.mean():.6f}, std={dit_out.std():.6f}")
    print(f"  attn (gated): mean={attn5.mean():.6f}")

    # ===========================================
    # Level 6: CFG ODE step (batch=2, DiTBlock + split + CFG scaling)
    # GIVEN: x [D, SEQ, 1], mu_in [D, SEQ, 2] (batch 1 = zero), t_emb, all block 0 weights
    # WHEN:  x_in = repeat(x, batch=2)
    #        out = DiTBlock(x_in, t_emb, pos_ids)  → [B, SEQ, D] batch=2
    #        [dphi, cfg_dphi] = split(out, batch dim)
    #        dphi = (1+cfg_rate)*dphi - cfg_rate*cfg_dphi
    #        x_new = x + dt * dphi
    # THEN:  x_new matches PyTorch
    # ===========================================
    print("\n--- Level 6: CFG ODE step (batch=2) ---")

    # Read inference_cfg_rate from model metadata
    cfg_rate = 0.7  # default, check GGUF metadata
    for kv in reader.fields.values():
        if kv.name == "decoder.inference_cfg_rate":
            cfg_rate = float(kv.parts[-1][0])
    print(f"  inference_cfg_rate={cfg_rate}")

    torch.manual_seed(45)
    cfg_x = torch.randn(1, SEQ, D)  # state [B=1, SEQ, D]

    # Repeat to batch=2
    cfg_x_in = cfg_x.repeat(2, 1, 1)  # [2, SEQ, D]

    # Prepare conditioning: batch 0 = real values, batch 1 = zeros (unconditional)
    # For this test we use the same random embedding as time_emb
    cfg_emb = te_out.clone().detach()  # [D]
    cfg_pos = torch.arange(SEQ).unsqueeze(0).expand(2, -1)  # [2, SEQ]

    # --- Run DiTBlock with batch=2 ---
    # AdaLayerNormZero (same emb for both batches — emb is not conditioned on batch)
    e_s = F_torch.silu(cfg_emb)
    e_p = adaln_linear(e_s)
    c6 = e_p.chunk(6, dim=-1)
    s_msa, sc_msa, g_msa = c6[0], c6[1] + 1.0, c6[2]
    s_mlp, sc_mlp, g_mlp = c6[3], c6[4] + 1.0, c6[5]

    xn6 = F_torch.layer_norm(cfg_x_in, [D], eps=1e-6)
    ain6 = xn6 * sc_msa.unsqueeze(0).unsqueeze(0) + s_msa.unsqueeze(0).unsqueeze(0)

    # Attention with batch=2
    q6 = to_q(ain6)
    k6 = to_k(ain6)
    v6 = to_v(ain6)
    q6 = q6.view(2, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    k6 = k6.view(2, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    v6 = v6.view(2, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    q6 = apply_rope(q6, cfg_pos, HEAD_DIM)
    k6 = apply_rope(k6, cfg_pos, HEAD_DIM)
    a6 = F_torch.scaled_dot_product_attention(q6, k6, v6)
    a6 = a6.transpose(1, 2).contiguous().view(2, SEQ, D)
    attn6 = to_out(a6)

    attn6 = attn6 * g_msa.unsqueeze(0).unsqueeze(0)
    x6_2 = cfg_x_in + attn6

    # FeedForward
    fn6 = F_torch.layer_norm(x6_2, [D], eps=1e-6)
    fn6 = fn6 * sc_mlp.unsqueeze(0).unsqueeze(0) + s_mlp.unsqueeze(0).unsqueeze(0)
    ff6 = F_torch.gelu(ff_up(fn6))
    ff6 = ff_down(ff6)
    ff6 = ff6 * g_mlp.unsqueeze(0).unsqueeze(0)
    dit6_out = x6_2 + ff6  # [2, SEQ, D]

    # Split batch dim: [0] = conditional, [1] = unconditional
    dphi_dt = dit6_out[0:1]      # [1, SEQ, D]
    cfg_dphi_dt = dit6_out[1:2]  # [1, SEQ, D]

    # CFG scaling: dphi = (1+r)*dphi - r*cfg_dphi
    cfg_dphi_scaled = cfg_dphi_dt * cfg_rate
    dphi_scaled = dphi_dt * (1.0 + cfg_rate)
    dphi_final = dphi_scaled - cfg_dphi_scaled  # [1, SEQ, D]

    # Euler step: x_new = x + dt * dphi
    t_span_1 = 1.0 - math.cos(0.1 * 0.5 * math.pi * 1)  # t_span[1]
    t_span_0 = 0.0  # t_span[0]
    dt_step1 = t_span_1 - t_span_0
    cfg_x_new = cfg_x + dt_step1 * dphi_final  # [1, SEQ, D]

    save_f32("level6_cfg_x", cfg_x[0])           # [SEQ, D]
    save_f32("level6_cfg_x_new", cfg_x_new[0])   # [SEQ, D]
    save_f32("level6_dphi_cond", dphi_dt[0])      # [SEQ, D] conditional output
    save_f32("level6_dphi_uncond", cfg_dphi_dt[0])# [SEQ, D] unconditional output
    save_f32("level6_dphi_final", dphi_final[0])  # [SEQ, D] after CFG

    print(f"  dphi_cond: mean={dphi_dt.mean():.6f}")
    print(f"  dphi_uncond: mean={cfg_dphi_dt.mean():.6f}")
    print(f"  dphi_final (after CFG): mean={dphi_final.mean():.6f}")
    print(f"  x_new: mean={cfg_x_new.mean():.6f}")
    print(f"  dt_step1={dt_step1:.6f}")

    print(f"\nAll gold references saved to {OUT}/")


if __name__ == "__main__":
    main()
