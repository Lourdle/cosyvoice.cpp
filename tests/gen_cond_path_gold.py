#!/usr/bin/env python3
"""Generate gold for conditioned InputEmbedding test.

Saves inputs in ggml layout [D, S] and output in flat [B*S*D] matching
both PyTorch C-contiguous [B, S, D] and ggml column-major [D, S, B].
"""
import math, os, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import gguf

GGUF_PATH = os.environ.get(
    "COSYVOICE_TEST_MODEL",
    os.path.join(os.path.dirname(__file__), "../models/tts/cosyvoice/CosyVoice3-2512_Q8_0.gguf"),
)
OUT = "/tmp/cv_cond_test_gold"
os.makedirs(OUT, exist_ok=True)

MEL = 80; D = 1024; SEQ = 20

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
    print(f"  {name}: shape={list(data.shape)}, numel={flat.size}, mean={flat.mean():.6f}")

def _dequant(t):
    ttype = t.tensor_type
    if ttype in (0, 1): return t.data.astype(np.float32).flatten()
    return gguf.dequantize(t.data, gguf.GGMLQuantizationType(ttype)).flatten()

reader = gguf.GGUFReader(GGUF_PATH)
cache = {t.name: t for t in reader.tensors}

# Load proj weight
t = cache['decoder.estimator.input_embed.proj.weight']
w_data = _dequant(t)
in_d, out_d = int(t.shape[0]), int(t.shape[1])
proj = nn.Linear(in_d, out_d)
proj.weight.data = torch.from_numpy(w_data.reshape(out_d, in_d).copy())
t_b = cache['decoder.estimator.input_embed.proj.bias']
proj.bias.data = torch.from_numpy(_dequant(t_b).copy())

# Deterministic inputs
torch.manual_seed(42)
x = torch.randn(1, SEQ, MEL)
cond = torch.randn(1, SEQ, MEL) * 2.5 - 5.32
mu = torch.randn(1, SEQ, MEL) * 1.0
spks = torch.randn(1, 1, MEL) * 0.117
spks_exp = spks.expand(1, SEQ, MEL)

# Save inputs as PyTorch C-contiguous [S, D] which matches ggml ne=[D, S] column-major
# (both have D varying fastest within each time step)
print("=== Inputs (PyTorch [S, D] = ggml ne=[D, S]) ===")
save_f32("x", x[0].contiguous().numpy())             # [20, 80]
save_f32("cond", cond[0].contiguous().numpy())       # [20, 80]
save_f32("mu", mu[0].contiguous().numpy())           # [20, 80]
save_f32("spks", spks[0].contiguous().numpy())       # [1, 80]

# Batch=2 CFG
x_b2 = x.repeat(2, 1, 1)
cond_b2 = torch.cat([cond, torch.zeros_like(cond)], dim=0)
mu_b2 = torch.cat([mu, torch.zeros_like(mu)], dim=0)
spks_b2 = torch.cat([spks_exp, torch.zeros(1, SEQ, MEL)], dim=0)

# InputEmbedding proj
ie_input = torch.cat([x_b2, cond_b2, mu_b2, spks_b2], dim=2)  # [2, SEQ, 320]
with torch.no_grad():
    ie_out = proj(ie_input)  # [2, SEQ, D]

print("\n=== InputEmbedding proj output ===")
# Save in ggml-compatible layout.
# ggml [D, S, B] flat data: for b in B, for s in S, for d in D → d + s*D + b*S*D
# PyTorch [B, S, D] C-contiguous: for b, for s, for d → d + s*D + b*S*D
# These are the same flat layout!
save_f32("ie_proj_output", ie_out.detach().numpy())  # [2, 20, 1024]

print(f"\n  batch0 mean={ie_out[0].mean():.6f}")
print(f"  batch1 mean={ie_out[1].mean():.6f}")
print(f"\nGold saved to {OUT}/")
