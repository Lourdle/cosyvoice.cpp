# Streaming TTS, DiT KV Cache & Buffer Policies

## Streaming TTS

Streaming TTS delivers audio chunks incrementally via a callback function as they are synthesized, without waiting for the full utterance to complete. This enables real-time playback and lower perceived latency.

Streaming is enabled via `--stream` flag on CLI/server. Chunk granularity is controlled by `--chunk-tokens`.

## DiT KV Cache

Each chunk runs the DiT's diffusion steps (10 by default) — without caching, every step of every new chunk recomputes attention over the whole sequence emitted so far. The **DiT KV cache** keeps the attention key/values of already-emitted positions **per diffusion step** (each step has its own cache slot), so a new chunk only computes its fresh positions: every cached position is processed once per step, then reused by all later chunks.

> The DiT KV cache is **only used during streaming TTS**; non-streaming calls ignore it.

### Slot Organization

The DiT KV cache is organized into **slots**, where each slot holds the KV cache for one diffusion step. With the default 10 diffusion steps, at most 10 slots exist (one per step).

Slots fall into three categories:

| Category | Memory | Behavior |
|----------|--------|----------|
| **Fixed** | Stays on device (GPU) | Fastest; never offloaded |
| **Offloadable** | Offloaded to CPU when not in use | Saves device memory at the cost of transfer latency |
| **Uncached** | Not stored at all | Full attention recomputation every step, no extra memory |

**Step-to-slot mapping.** Steps are laid out in order — uncached first, then offloadable (all sharing one device scratch slot, each with its own CPU buffer), then fixed (one dedicated device slot each). A configuration with exactly one offloadable slot is normalized to fixed (`offloadable=1 → fixed+1`), and both counts are clamped so `fixed + offloadable` never exceeds the diffusion step count. Total cached steps = `fixed + offloadable`; the rest recompute fully. Internal slot/scheduling layout: [API_cosyvoice.md — DiT KV Cache Concept](API_cosyvoice.md#dit-kv-cache-concept).

Adjacent steps can also **share** one physical cache via `--dit-kv-actual-fixed-slots` / `--dit-kv-actual-offloadable-slots`: the fixed and offloadable steps are each partitioned into that many groups, one device slot / CPU buffer per group, so the cache shrinks to `actual / nominal` of its size. Steps inside a group read the history last written by the group (a small approximation), and CPU round-trips happen once per group instead of per step.

The cache is large, so the default is 0 slots (all 10 steps fully recomputed). When enabled and the sequence exceeds the configured cache length, some positions are discarded — inference continues normally but output quality may degrade. Offloadable slots transfer data between device and CPU, which may not improve speed and can be slower than full recomputation depending on bandwidth.

### Configuration

DiT KV cache parameters are configured via CLI/server `--dit-kv-*` flags (context parameters — on the CLI they apply to interactive/streaming mode):

- `--dit-kv-cache-type`: Storage format for the cache — `f32`/`f16`/`q8_0`/`q5_1`/`q5_0`/`q4_1`/`q4_0`, or asymmetric with separate K and V formats (`k=<type>,v=<type>[,fallback=<type>]`, same style as `--llm-kv-cache-type`).
- `--dit-kv-fixed-slots`: Number of device-resident slots. Default: `0`.
- `--dit-kv-offloadable-slots`: Number of CPU-offloadable slots. Default: `0`.
- `--dit-kv-actual-fixed-slots`: Physical device slots backing the fixed slots (adjacent steps share one). `0` = no sharing. Default: `0`.
- `--dit-kv-actual-offloadable-slots`: Physical CPU buffers backing the offloadable slots (adjacent steps share one). `0` = no sharing. Default: `0`.
- `--dit-kv-cache-length`: Maximum sequence positions kept in the cache. Default: `0` = max LLM length × 10.

The number of diffusion steps is read from the model's `decoder.diffusion_steps` GGUF metadata (defaulting to 10 when absent). It can be overridden at load time via `cosyvoice_context_params_v4_t::diffusion_steps` — `--diffusion-steps` on the CLI/server (pass `<= 0` to keep the metadata value); any value is clamped to the runtime maximum of 50. Query the effective count with `cosyvoice_get_diffusion_steps()`.

Suggested starting points (10 diffusion steps total):

| Goal | `--dit-kv-fixed-slots` | `--dit-kv-offloadable-slots` |
|------|------------------------|------------------------------|
| Disable cache (default) | `0` | `0` |
| Fastest, plenty of device memory | `10` | `0` |
| Balanced speed / memory | `2` | `0` |
| Speed with reduced device memory | `1` | `9` |
| Minimal device memory (slower) | `0` | `9` |

## Inference Buffer Policies

The inference engine uses a buffer policy that controls how intermediate tensors are allocated (default: `dedicated`; the non-interactive one-shot CLI path always uses `shared`):

- `shared`: LLM KV cache shares memory with DiT intermediate buffers. Each inference runs the LLM module fully. Saves memory but can cause instability on CUDA when Flash Attention is disabled.
- `balanced`: Like `shared`, but offloads reusable LLM KV cache to CPU after LLM inference completes.
- `dedicated`: Fully independent buffers. LLM KV cache persists on device and can be reused quickly across steps — recommended for streaming.

The buffer policy is a context parameter, configurable via `cosyvoice_context_params_v4_t` (`--inference-buffer-policy` on the CLI/server / WebUI).
