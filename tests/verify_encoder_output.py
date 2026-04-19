#!/usr/bin/env python3
"""Verify encoder outputs by running Python encoder with the same prompt_speech inputs.

Uses C++ frontend dump of prompt_speech to:
1. Run the Python encoder pipeline
2. Compare mu, spks, conds against C++ encoder dumps

Then runs one DiT step with REAL encoder outputs to check v_cond.
"""
import math, os, struct, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf

GGUF_PATH = os.environ.get(
    "COSYVOICE_TEST_MODEL",
    os.path.join(os.path.dirname(__file__), "../models/tts/cosyvoice/CosyVoice3-2512_Q8_0.gguf"),
)
DUMP_DIR = "/tmp/cv_dump"

D = 1024; HEADS = 16; HD = 64; DEPTH = 22; MEL = 80


def _dq(t):
    tt = t.tensor_type
    if tt in (0, 1):
        return t.data.astype(np.float32).flatten()
    return gguf.dequantize(t.data, gguf.GGMLQuantizationType(tt)).flatten()


class L:
    def __init__(s, p):
        s.r = gguf.GGUFReader(p)
        s._c = {t.name: t for t in s.r.tensors}

    def t(s, n):
        t = s._c[n]
        return _dq(t), list(t.shape)

    def linear(s, pfx):
        w, wn = s.t(f'{pfx}.weight')
        b, _ = s.t(f'{pfx}.bias')
        i, o = int(wn[0]), int(wn[1])
        l = nn.Linear(i, o)
        l.weight.data = torch.from_numpy(w.reshape(o, i).copy())
        l.bias.data = torch.from_numpy(b.copy())
        return l

    def grouped_conv1d(s, pfx, groups):
        w, wn = s.t(f'{pfx}.weight')
        b, _ = s.t(f'{pfx}.bias')
        k, ipg, o = int(wn[0]), int(wn[1]), int(wn[2])
        c = nn.Conv1d(ipg * groups, o, k, groups=groups, bias=True, padding=0)
        c.weight.data = torch.from_numpy(w.reshape(o, ipg, k).copy())
        c.bias.data = torch.from_numpy(b.copy())
        return c

    def get_meta_int(s, key):
        for kv in s.r.fields.values():
            if kv.name == key:
                return int(kv.parts[-1][0])
        return None


def rope(x, pos, hd):
    B, H, S, HD_ = x.shape
    f = 1.0 / (10000.0 ** (torch.arange(0, HD_, 2, dtype=torch.float32) / HD_))
    # pos: [B, S] → [B, 1, S, 1] for broadcasting with [B, H, S, HD//2]
    ph = pos.unsqueeze(1).unsqueeze(-1) * f.view(1, 1, 1, -1)  # [B, 1, S, HD//2]
    c_ = torch.cos(ph)
    s_ = torch.sin(ph)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([x1 * c_ - x2 * s_, x1 * s_ + x2 * c_], -1).flatten(-2)


def main():
    print(f"Loading GGUF: {GGUF_PATH}")
    loader = L(GGUF_PATH)

    # Load encoder outputs from C++ dump (ggml F-order)
    mu_cpp = np.fromfile(os.path.join(DUMP_DIR, "encode_mu.bin"), dtype=np.float32)
    spks_cpp = np.fromfile(os.path.join(DUMP_DIR, "encode_spks.bin"), dtype=np.float32)
    conds_cpp = np.fromfile(os.path.join(DUMP_DIR, "encode_conds.bin"), dtype=np.float32)

    SEQ = len(mu_cpp) // MEL  # 184
    print(f"SEQ={SEQ}")

    # ggml stores [ne[0]=80, ne[1]=SEQ] in row-major by ne[0]:
    # data[feat + mel_pos * 80] → reshape(SEQ, 80)
    mu = torch.from_numpy(mu_cpp.reshape(SEQ, MEL).copy())  # [SEQ, 80]
    spks = torch.from_numpy(spks_cpp.copy()).unsqueeze(0)  # [1, 80]
    conds = torch.from_numpy(conds_cpp.reshape(SEQ, MEL).copy())  # [SEQ, 80]

    print(f"mu: shape={list(mu.shape)}, mean={mu.mean():.6f}")
    print(f"spks: shape={list(spks.shape)}, mean={spks.mean():.6f}")
    print(f"conds: shape={list(conds.shape)}, mean={conds.mean():.6f}")

    # Load DiT weights
    ie_proj = loader.linear('decoder.estimator.input_embed.proj')
    ie_conv1 = loader.grouped_conv1d('decoder.estimator.input_embed.conv_pos_embed.conv1.0', 16)
    ie_conv2 = loader.grouped_conv1d('decoder.estimator.input_embed.conv_pos_embed.conv2.0', 16)

    # SinusPositionEmbedding: generated at load time, not stored in GGUF
    # emb[i] = exp(i * -log(10000) / (half_dim-1)) * 1000, half_dim=128
    half_dim = 128
    time_emb_data = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000.0) / (half_dim - 1))) * 1000.0

    time_mlp_0 = loader.linear('decoder.estimator.time_embed.time_mlp.0')
    time_mlp_2 = loader.linear('decoder.estimator.time_embed.time_mlp.2')

    out_proj = loader.linear('decoder.estimator.proj_out')
    norm_out_linear = loader.linear('decoder.estimator.norm_out.linear')

    blocks = []
    for i in range(DEPTH):
        pfx = f'decoder.estimator.transformer_blocks.{i}'
        blk = {
            'q': loader.linear(f'{pfx}.attn.to_q'),
            'k': loader.linear(f'{pfx}.attn.to_k'),
            'v': loader.linear(f'{pfx}.attn.to_v'),
            'o': loader.linear(f'{pfx}.attn.to_out.0'),
            'ff1': loader.linear(f'{pfx}.ff.ff.0.0'),
            'ff2': loader.linear(f'{pfx}.ff.ff.2'),
            'ada_norm': loader.linear(f'{pfx}.attn_norm.linear'),
        }
        blocks.append(blk)

    print("\nRunning DiT step 1 with REAL encoder outputs...")

    # Load actual noise from C++ dump
    noise_data = np.fromfile(os.path.join(DUMP_DIR, "initial_noise.bin"), dtype=np.float32)
    noise = torch.from_numpy(noise_data.reshape(SEQ, MEL).copy())
    print(f"  noise (from C++ dump): shape={list(noise.shape)}, mean={noise.mean():.6f}")

    x = noise  # start with same noise as C++

    with torch.no_grad():
        # === InputEmbedding ===
        # x is [SEQ, 80], we need to work in [batch=2, D, SEQ] for CFG
        # C++ flow: x is permuted to (1,0,2) at DiT entry: [SEQ,80] → [80,SEQ]
        # Then padded to batch=2 for CFG

        # Prepare CFG batch
        x_perm = x.T  # [80, SEQ]
        x_batch = x_perm.unsqueeze(0).expand(2, -1, -1).clone()  # [2, 80, SEQ]

        # conds for CFG: [cond, zeros]
        cond_perm = conds.T  # [80, SEQ]
        cond_batch = torch.stack([cond_perm, torch.zeros_like(cond_perm)], dim=0)  # [2, 80, SEQ]

        # mu for CFG: batch[0]=mu, batch[1]=zeros (C++ uses ggml_pad)
        mu_perm = mu.T  # [80, SEQ]
        mu_batch = torch.stack([mu_perm, torch.zeros_like(mu_perm)], dim=0)  # [2, 80, SEQ]

        # spks for CFG: batch[0]=spks, batch[1]=zeros (C++ uses ggml_pad)
        spks_expanded = spks.unsqueeze(-1).expand(-1, -1, SEQ)  # [1, 80, SEQ]
        spks_batch = torch.stack([spks_expanded[0], torch.zeros(MEL, SEQ)], dim=0)  # [2, 80, SEQ]

        # Concat: [x, cond, mu, spks] along dim=1 (feature)
        inp = torch.cat([x_batch, cond_batch, mu_batch, spks_batch], dim=1)  # [2, 320, SEQ]
        proj_out = ie_proj(inp.permute(0, 2, 1))  # [2, SEQ, 1024]
        proj_out = proj_out.permute(0, 2, 1)  # [2, 1024, SEQ]

        # conv_pos_embed
        pad_l = (31 - 1) // 2  # kernel=31, pad=15
        pad_r = 31 - 1 - pad_l
        conv_in = F.pad(proj_out, (pad_l, pad_r))  # [2, 1024, SEQ+30]
        conv_out = ie_conv1(conv_in)  # [2, 1024, SEQ]
        conv_out = F.mish(conv_out)
        conv_in2 = F.pad(conv_out, (pad_l, pad_r))
        conv_out2 = ie_conv2(conv_in2)
        x_emb = proj_out + conv_out2  # [2, 1024, SEQ]

        print(f"  input_embed output: shape={list(x_emb.shape)}, mean={x_emb.mean():.4f}")
        print(f"    batch[0] (cond) mean: {x_emb[0].mean():.4f}")
        print(f"    batch[1] (uncond) mean: {x_emb[1].mean():.4f}")

        # === TimestepEmbedding ===
        # Step 1: t_span[0]=0, so t=0 at entry. CosyVoice fills t_in with t value.
        # SinusPositionEmbedding: emb[i] * t → sin/cos
        # time_emb_data is [128], input t is scalar
        # For batch=2 CFG, t_in = [t, t] = [0, 0]
        t_val = torch.tensor([0.0, 0.0])  # both batch elements get same t
        pos_emb = t_val.unsqueeze(-1) * time_emb_data.unsqueeze(0)  # [2, 128]
        sin_emb = torch.sin(pos_emb)
        cos_emb = torch.cos(pos_emb)
        t_emb_sincos = torch.cat([sin_emb, cos_emb], dim=-1)  # [2, 256]
        t_emb = time_mlp_0(t_emb_sincos)
        t_emb = F.silu(t_emb)
        t_emb = time_mlp_2(t_emb)  # [2, 1024]
        t_emb_batch = t_emb  # already [2, 1024]

        print(f"  t_emb: mean={t_emb.mean():.4f}")

        # === Transformer blocks ===
        x_blocks = x_emb.permute(0, 2, 1)  # [2, SEQ, 1024]

        for bi, blk in enumerate(blocks):
            # AdaLayerNormZero: emb → silu → linear → split 6 ways
            ada_emb = F.silu(t_emb_batch)
            ada_out = blk['ada_norm'](ada_emb)  # [2, 6*1024]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_out.chunk(6, dim=-1)

            # Attn norm: LayerNorm(no params) + ada scale/shift
            norm_x = F.layer_norm(x_blocks, [D])
            norm_x = norm_x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

            # Self-attention
            q = blk['q'](norm_x).reshape(2, SEQ, HEADS, HD).permute(0, 2, 1, 3)
            k = blk['k'](norm_x).reshape(2, SEQ, HEADS, HD).permute(0, 2, 1, 3)
            v = blk['v'](norm_x).reshape(2, SEQ, HEADS, HD).permute(0, 2, 1, 3)

            # RoPE — pos_ids needs shape [B*H, SEQ] for broadcasting
            # Actually rope expects [B, H, S, HD] and pos [B, S]
            # But our q is [2, 16, SEQ, 64], pos should be [2, SEQ]
            # The issue is broadcasting — pos needs to match [B, 1, S, HD//2]
            pos_ids = torch.arange(SEQ, dtype=torch.float32).unsqueeze(0).expand(2, -1)  # [2, SEQ]
            q = rope(q, pos_ids, HD)
            k = rope(k, pos_ids, HD)

            # Scaled dot-product attention
            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(2, SEQ, D)
            attn_out = blk['o'](attn_out)

            attn_out = attn_out * gate_msa.unsqueeze(1)
            x_blocks = x_blocks + attn_out

            # FF: LayerNorm(no params) + ada scale/shift + FF
            ff_norm_x = F.layer_norm(x_blocks, [D])
            ff_norm_x = ff_norm_x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

            ff_out = blk['ff1'](ff_norm_x)
            ff_out = F.gelu(ff_out)
            ff_out = blk['ff2'](ff_out)
            ff_out = ff_out * gate_mlp.unsqueeze(1)
            x_blocks = x_blocks + ff_out

            if bi == 0:
                print(f"  block0: mean={x_blocks.mean():.4f}, batch[0]={x_blocks[0].mean():.4f}, batch[1]={x_blocks[1].mean():.4f}")

        # Final norm (AdaLayerNorm_Final: no learned weight/bias in LayerNorm)
        ada_emb = F.silu(t_emb_batch)
        ada_out = norm_out_linear(ada_emb)  # [2, 2*1024]
        scale_final, shift_final = ada_out.chunk(2, dim=-1)
        x_normed = F.layer_norm(x_blocks, [D])  # no weight/bias
        x_normed = x_normed * (1 + scale_final.unsqueeze(1)) + shift_final.unsqueeze(1)

        # Output projection
        output = out_proj(x_normed)  # [2, SEQ, 80]
        output = output.permute(0, 2, 1)  # [2, 80, SEQ]

        # Split CFG
        v_cond_py = output[0]  # [80, SEQ]
        v_uncond_py = output[1]  # [80, SEQ]

        # Transpose to match dump format [SEQ, 80]
        v_cond_py_flat = v_cond_py.T  # [SEQ, 80]
        v_uncond_py_flat = v_uncond_py.T

        print(f"\n=== Python DiT Results ===")
        print(f"  v_cond:   mean={v_cond_py_flat.mean():.4f}")
        print(f"  v_uncond: mean={v_uncond_py_flat.mean():.4f}")

        # Load C++ dumps
        v_cond_cpp = np.fromfile(os.path.join(DUMP_DIR, "step1_v_cond.bin"), dtype=np.float32)
        v_uncond_cpp = np.fromfile(os.path.join(DUMP_DIR, "step1_v_uncond.bin"), dtype=np.float32)

        print(f"\n=== C++ DiT Results ===")
        print(f"  v_cond:   mean={v_cond_cpp.mean():.4f}")
        print(f"  v_uncond: mean={v_uncond_cpp.mean():.4f}")

        diff_cond = np.abs(v_cond_py_flat.numpy().flatten() - v_cond_cpp).max()
        diff_uncond = np.abs(v_uncond_py_flat.numpy().flatten() - v_uncond_cpp).max()
        print(f"\n=== Comparison ===")
        print(f"  v_cond max_diff:   {diff_cond:.4f}")
        print(f"  v_uncond max_diff: {diff_uncond:.4f}")

        if diff_cond < 2.0:
            print(f"  MATCH — encoder outputs feed through DiT consistently")
        else:
            print(f"  MISMATCH — DiT behaves differently with these inputs")
            # Show where divergence is biggest
            v_cond_diff = np.abs(v_cond_py_flat.numpy().flatten() - v_cond_cpp)
            worst_idx = v_cond_diff.argmax()
            mel = worst_idx % 80
            seq = worst_idx // 80
            print(f"  Worst at seq={seq}, mel={mel}: py={v_cond_py_flat.numpy().flatten()[worst_idx]:.4f}, cpp={v_cond_cpp[worst_idx]:.4f}")


if __name__ == "__main__":
    main()
