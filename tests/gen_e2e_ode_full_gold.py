#!/usr/bin/env python3
"""Generate gold for full 10-step ODE: steps 2-10 from step1_feat.

Runs two patterns:
  A) correct  — each step uses the correct t/dt from cosine schedule
  B) prodmatch — step 2 uses step 1's t/dt (reproduces cosyvoice-tts.cpp:358 bug)

Requires: /tmp/cv_repro3/encode_mu.bin and step1_feat.bin

Output: /tmp/cv_e2e_ode_full_gold/
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
OUT = "/tmp/cv_e2e_ode_full_gold"
os.makedirs(OUT, exist_ok=True)

D = 1024
HEADS = 16
HD = D // HEADS
DEPTH = 22
MEL_DIM = 80
HALF_DIM = 128
SUB_SEQ = 20


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
        sh_m, sc_m, g_m = ch[0], ch[1] + 1, ch[2]
        sh_f, sc_f, g_f = ch[3], ch[4] + 1, ch[5]

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


def run_ode_step(x_pt, mu_pt, t_val, dt_val, blocks, ie_proj, time_mlp_0, time_mlp_2,
                 norm_out_linear, proj_out, emb_coeff, cfg_rate):
    """Run a single ODE Euler step through the full pipeline. Returns (x_new, dphi_final)."""
    with torch.no_grad():
        # Timestep embedding
        sinpos = torch.cat([
            torch.sin(torch.tensor([t_val]) * emb_coeff),
            torch.cos(torch.tensor([t_val]) * emb_coeff),
        ])
        te = time_mlp_2(F_torch.silu(time_mlp_0(sinpos)))

        # Batch=2 CFG
        x_b2 = x_pt.repeat(2, 1, 1)
        mu_b2 = torch.cat([mu_pt, torch.zeros_like(mu_pt)], dim=0)
        spks_b2 = torch.zeros(2, SUB_SEQ, MEL_DIM)
        cond_b2 = torch.zeros(2, SUB_SEQ, MEL_DIM)

        ie_input = torch.cat([x_b2, cond_b2, mu_b2, spks_b2], dim=2)
        ie_out = ie_proj(ie_input)

        # 22 DiT blocks
        pos_ids = torch.arange(SUB_SEQ).unsqueeze(0).expand(2, -1)
        h = ie_out
        for block in blocks:
            h = block.forward(h, te, pos_ids)

        # AdaLayerNorm_Final + proj_out
        e_f = norm_out_linear(F_torch.silu(te))
        scale_f, shift_f = e_f.chunk(2, dim=-1)
        scale_f = scale_f + 1
        h = F_torch.layer_norm(h, [D], eps=1e-6)
        h = h * scale_f.unsqueeze(0).unsqueeze(0) + shift_f.unsqueeze(0).unsqueeze(0)
        dit_out = proj_out(h)

        # CFG
        dphi_cond = dit_out[0:1]
        dphi_uncond = dit_out[1:2]
        dphi_final = (1 + cfg_rate) * dphi_cond - cfg_rate * dphi_uncond

        # Euler step
        x_new = x_pt + dt_val * dphi_final

    return x_new, dphi_final


def main():
    repro_dir = "/tmp/cv_repro3"
    if not os.path.exists(os.path.join(repro_dir, "step1_feat.bin")):
        print(f"ERROR: {repro_dir}/step1_feat.bin not found. Run voice clone first.")
        sys.exit(1)

    step1_data = np.fromfile(os.path.join(repro_dir, "step1_feat.bin"), dtype=np.float32)
    encode_mu = np.fromfile(os.path.join(repro_dir, "encode_mu.bin"), dtype=np.float32)

    seq_len = len(step1_data) // MEL_DIM
    print(f"step1: {len(step1_data)} floats, seq_len={seq_len}, SUB_SEQ={SUB_SEQ}")

    x_np = step1_data.reshape(MEL_DIM, seq_len).T[:SUB_SEQ, :]
    mu_np = encode_mu.reshape(MEL_DIM, seq_len).T[:SUB_SEQ, :]
    mu_pt = torch.from_numpy(mu_np.copy()).unsqueeze(0)

    # Save initial state for C++ test
    save_f32("initial_x", x_np)
    save_f32("initial_mu", mu_np)

    print(f"\nLoading GGUF: {GGUF_PATH}")
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

    cfg_rate = 0.7
    for kv in loader.reader.fields.values():
        if kv.name == "decoder.inference_cfg_rate":
            cfg_rate = float(kv.parts[-1][0])
    print(f"cfg_rate={cfg_rate}")

    # Cosine schedule (11 points, 10 steps)
    t_span = [1.0 - math.cos(0.1 * 0.5 * math.pi * i) for i in range(11)]
    print(f"t_span = {[f'{v:.6f}' for v in t_span]}")

    # get_t_and_dt matching C++ logic
    def get_t_and_dt(step):
        t = t_span[0]
        for cur in range(1, step):
            t += t_span[cur] - t_span[cur - 1]
        dt = t_span[step] - t_span[step - 1]
        return t, dt

    common_args = dict(
        blocks=blocks, ie_proj=ie_proj, time_mlp_0=time_mlp_0, time_mlp_2=time_mlp_2,
        norm_out_linear=norm_out_linear, proj_out=proj_out, emb_coeff=emb_coeff,
        cfg_rate=cfg_rate,
    )

    summary_lines = []

    # --- Pattern A: correct t/dt ---
    print("\n=== Pattern A: correct t/dt ===")
    x_pt = torch.from_numpy(x_np.copy()).unsqueeze(0)
    for step in range(2, 11):
        t_val, dt_val = get_t_and_dt(step)
        x_pt, dphi = run_ode_step(x_pt, mu_pt, t_val, dt_val, **common_args)
        save_f32(f"correct_step{step:02d}_x", x_pt[0])
        save_f32(f"correct_step{step:02d}_dphi", dphi[0])
        line = f"  correct step{step:2d}: t={t_val:.6f} dt={dt_val:.6f} x_mean={x_pt.mean():.6f} dphi_mean={dphi.mean():.6f}"
        print(line)
        summary_lines.append(line)

    # --- Pattern B: prodmatch (step2 uses step=1 t/dt) ---
    print("\n=== Pattern B: prodmatch (step2 bug) ===")
    x_pt = torch.from_numpy(x_np.copy()).unsqueeze(0)
    for step in range(2, 11):
        if step == 2:
            # BUG: cosyvoice-tts.cpp:358 calls build_cgraph_one_step with step=1
            t_val, dt_val = get_t_and_dt(1)
        else:
            t_val, dt_val = get_t_and_dt(step)
        x_pt, dphi = run_ode_step(x_pt, mu_pt, t_val, dt_val, **common_args)
        save_f32(f"prodmatch_step{step:02d}_x", x_pt[0])
        save_f32(f"prodmatch_step{step:02d}_dphi", dphi[0])
        line = f"  prodmatch step{step:2d}: t={t_val:.6f} dt={dt_val:.6f} x_mean={x_pt.mean():.6f} dphi_mean={dphi.mean():.6f}"
        print(line)
        summary_lines.append(line)

    # Save summary
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nAll gold saved to {OUT}/")


if __name__ == "__main__":
    main()
