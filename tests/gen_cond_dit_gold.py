#!/usr/bin/env python3
"""Generate gold for full DiT with non-zero conditioning, batch=2 CFG.

Self-contained: uses synthetic deterministic inputs (no dump files needed).
Generates ie_proj output and per-block outputs for progressive comparison.
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
OUT = os.environ.get("COSYVOICE_TEST_GOLD_DIR", "/tmp/cv_cond_dit_gold")
os.makedirs(OUT, exist_ok=True)

D = 1024; HEADS = 16; HD = 64; DEPTH = 22; MEL = 80; HALF = 128
SEQ = int(os.environ.get("COSYVOICE_TEST_SEQ", "20")); BATCH = 2

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
    return flat.mean()

def _dq(t):
    tt = t.tensor_type
    if tt in (0,1): return t.data.astype(np.float32).flatten()
    return gguf.dequantize(t.data, gguf.GGMLQuantizationType(tt)).flatten()

class L:
    def __init__(s,p):
        s.r = gguf.GGUFReader(p)
        s._c = {t.name:t for t in s.r.tensors}
    def t(s,n): t=s._c[n]; return _dq(t),list(t.shape)
    def linear(s,pfx):
        w,wn=s.t(f'{pfx}.weight'); b,_=s.t(f'{pfx}.bias')
        i,o=int(wn[0]),int(wn[1])
        l=nn.Linear(i,o); l.weight.data=torch.from_numpy(w.reshape(o,i).copy())
        l.bias.data=torch.from_numpy(b.copy()); return l
    def grouped_conv1d(s,pfx,groups):
        w,wn=s.t(f'{pfx}.weight'); b,_=s.t(f'{pfx}.bias')
        k,ipg,o=int(wn[0]),int(wn[1]),int(wn[2])
        c=nn.Conv1d(ipg*groups,o,k,groups=groups,bias=True,padding=0)
        c.weight.data=torch.from_numpy(w.reshape(o,ipg,k).copy())
        c.bias.data=torch.from_numpy(b.copy()); return c

def rope(x,pos,hd):
    B,H,S,HD_=x.shape
    f=1.0/(10000.0**(torch.arange(0,HD_,2,dtype=torch.float32)/HD_))
    p=pos.unsqueeze(-1).float()
    a=p*f.unsqueeze(0).unsqueeze(0)
    c,s=torch.cos(a).unsqueeze(1),torch.sin(a).unsqueeze(1)
    x0,x1=x[...,0::2],x[...,1::2]
    return torch.stack([x0*c-x1*s,x0*s+x1*c],dim=-1).flatten(-2)

class Block:
    def __init__(s,l,i):
        p=f'decoder.estimator.transformer_blocks.{i}'
        s.a=l.linear(f'{p}.attn_norm.linear')
        s.q=l.linear(f'{p}.attn.to_q'); s.k=l.linear(f'{p}.attn.to_k')
        s.v=l.linear(f'{p}.attn.to_v'); s.o=l.linear(f'{p}.attn.to_out.0')
        s.fu=l.linear(f'{p}.ff.ff.0.0'); s.fd=l.linear(f'{p}.ff.ff.2')
    @torch.no_grad()
    def fwd(s,x,e,p):
        B,S,D_=x.shape
        ch=s.a(F.silu(e)).chunk(6,dim=-1)
        sm,cm,gm=ch[0],ch[1]+1,ch[2]; sf,cf,gf=ch[3],ch[4]+1,ch[5]
        n=F.layer_norm(x,[D_],eps=1e-6)
        n=n*cm.unsqueeze(0).unsqueeze(0)+sm.unsqueeze(0).unsqueeze(0)
        q=s.q(n).view(B,S,HEADS,HD).transpose(1,2)
        k=s.k(n).view(B,S,HEADS,HD).transpose(1,2)
        v=s.v(n).view(B,S,HEADS,HD).transpose(1,2)
        q=rope(q,p,HD); k=rope(k,p,HD)
        a=F.scaled_dot_product_attention(q,k,v)
        a=a.transpose(1,2).contiguous().view(B,S,D_)
        a=s.o(a)*gm.unsqueeze(0).unsqueeze(0)
        x=x+a
        fn=F.layer_norm(x,[D_],eps=1e-6)
        fn=fn*cf.unsqueeze(0).unsqueeze(0)+sf.unsqueeze(0).unsqueeze(0)
        ff=s.fd(F.gelu(s.fu(fn)))
        return x+ff*gf.unsqueeze(0).unsqueeze(0)

def mish(x):
    return x * torch.tanh(F.softplus(x))

print(f"Loading {GGUF_PATH}...")
l = L(GGUF_PATH)
blocks = [Block(l,i) for i in range(DEPTH)]
no = l.linear('decoder.estimator.norm_out.linear')
po = l.linear('decoder.estimator.proj_out')
ie = l.linear('decoder.estimator.input_embed.proj')
t0 = l.linear('decoder.estimator.time_embed.time_mlp.0')
t2 = l.linear('decoder.estimator.time_embed.time_mlp.2')
ec = torch.exp(torch.arange(HALF,dtype=torch.float32)*(-math.log(10000)/(HALF-1)))*1000.0
ts = [1.0-math.cos(0.1*0.5*math.pi*i) for i in range(11)]

# conv_pos_embed weights (grouped conv, groups=16)
cpe_conv1 = l.grouped_conv1d('decoder.estimator.input_embed.conv_pos_embed.conv1.0', 16)
cpe_conv2 = l.grouped_conv1d('decoder.estimator.input_embed.conv_pos_embed.conv2.0', 16)
CPE_KERNEL = cpe_conv1.weight.shape[2]
print(f"conv_pos_embed: kernel={CPE_KERNEL}, groups=16")

# Deterministic inputs
torch.manual_seed(42)
x = torch.randn(1, SEQ, MEL)
cond = torch.randn(1, SEQ, MEL) * 2.5 - 5.32
mu = torch.randn(1, SEQ, MEL) * 1.0
spks = torch.randn(1, 1, MEL) * 0.117

# Save inputs (PyTorch C-contiguous layout = ggml column-major)
print("=== Inputs ===")
save_f32("x", x[0].numpy())
save_f32("cond", cond[0].numpy())
save_f32("mu", mu[0].numpy())
save_f32("spks", spks[0].numpy())

# Batch=2 CFG
xb = x.repeat(2,1,1)
cb = torch.cat([cond, torch.zeros_like(cond)], dim=0)
mb = torch.cat([mu, torch.zeros_like(mu)], dim=0)
sb = torch.cat([spks.expand(1,SEQ,MEL), torch.zeros(1,SEQ,MEL)], dim=0)

# Timestep
t = ts[0]
with torch.no_grad():
    sp = torch.cat([torch.sin(torch.tensor([t])*ec), torch.cos(torch.tensor([t])*ec)])
    te = t2(F.silu(t0(sp)))

save_f32("time_emb", te.numpy())

# InputEmbedding proj
ie_in = torch.cat([xb,cb,mb,sb], dim=2)
with torch.no_grad():
    h_proj = ie(ie_in)

m = save_f32("ie_proj", h_proj.numpy())
print(f"  ie_proj: mean={m:.6f}, b0={h_proj[0].mean():.6f}, b1={h_proj[1].mean():.6f}")

# conv_pos_embed: [B, SEQ, D] → transpose → [B, D, SEQ] → causal conv → transpose back
with torch.no_grad():
    h_conv = h_proj.transpose(1, 2)  # [2, 1024, 20]

    # conv1
    h_conv = F.pad(h_conv, (CPE_KERNEL - 1, 0))
    h_conv = cpe_conv1(h_conv)
    save_f32("cpe_conv1_out", h_conv.transpose(1,2).numpy())
    h_conv = mish(h_conv)
    save_f32("cpe_conv1_mish", h_conv.transpose(1,2).numpy())

    # conv2
    h_conv = F.pad(h_conv, (CPE_KERNEL - 1, 0))
    h_conv_pre_conv2 = h_conv.clone()
    h_conv = cpe_conv2(h_conv)
    save_f32("cpe_conv2_out", h_conv.transpose(1,2).numpy())
    h_conv = mish(h_conv)
    save_f32("cpe_conv2_mish", h_conv.transpose(1,2).numpy())

    h_conv = h_conv.transpose(1, 2)  # back to [B, SEQ, D]
    h = h_proj + h_conv  # residual

m = save_f32("ie_full", h.numpy())
print(f"  ie_full (proj+conv_pos_embed): mean={m:.6f}, b0={h[0].mean():.6f}, b1={h[1].mean():.6f}")
print(f"  conv_pos_embed delta: {(h-h_proj).mean():.6f}, b0={((h-h_proj)[0]).mean():.6f}, b1={((h-h_proj)[1]).mean():.6f}")

# DiT blocks (using ie_full = proj + conv_pos_embed as input)
pid = torch.arange(SEQ).unsqueeze(0).expand(2,-1)
with torch.no_grad():
    for i,b in enumerate(blocks):
        h = b.fwd(h, te, pid)  # h starts as ie_full
        m = save_f32(f"block{i}", h.numpy())
        if i < 3 or i == DEPTH-1:
            print(f"  block{i}: mean={m:.6f}, b0={h[0].mean():.6f}, b1={h[1].mean():.6f}")

    # Final norm + proj_out
    ef = no(F.silu(te))
    sf,hf = ef.chunk(2, dim=-1)
    h = F.layer_norm(h, [D], eps=1e-6)
    h = h * (sf+1).unsqueeze(0).unsqueeze(0) + hf.unsqueeze(0).unsqueeze(0)
    dit_out = po(h)

vc = dit_out[0:1]; vu = dit_out[1:2]
# Save as [B, SEQ, MEL] C-contiguous — matches ggml [MEL, SEQ, B] flat layout
save_f32("v_cond", vc.numpy())   # [1, SEQ, MEL]
save_f32("v_uncond", vu.numpy()) # [1, SEQ, MEL]

# CFG formula: (1 + cfg_rate) * v_cond - cfg_rate * v_uncond
cfg_rate = 0.7
for kv in l.r.fields.values():
    if kv.name == 'decoder.inference_cfg_rate':
        cfg_rate = float(kv.parts[-1][0])
v_cfg = (1 + cfg_rate) * vc - cfg_rate * vu
save_f32("v_cfg", v_cfg.numpy())  # [1, SEQ, MEL]

# Euler step: x + dt * v_cfg
dt = ts[1] - ts[0]
x_new = x + dt * v_cfg
save_f32("x_after_step1", x_new[0].numpy())

print(f"\n=== Result ===")
print(f"  v_cond={vc.mean():.4f}, v_uncond={vu.mean():.4f}, delta={vc.mean()-vu.mean():.4f}")
print(f"  v_cfg={v_cfg.mean():.4f}, dt={dt:.6f}")
print(f"  x_after_step1={x_new.mean():.4f}")
print(f"\nGold saved to {OUT}/")
