# LLM graph reuse regression

This model-backed test compares repeated single-token decoding (which reuses the
Flash Attention graph) with fresh full-prefix evaluations. It needs a CosyVoice 3
GGUF with non-F32 speech embeddings, such as Q4_K_M, but no reference recording,
ONNX frontend assets, or network access.

Configure with `-DCOSYVOICE_BUILD_TESTS=ON` and build `test-llm-graph-reuse`, then run:

```sh
build/bin/test-llm-graph-reuse /path/to/CosyVoice3-2512_Q4_K_M.gguf MTL0
build/bin/test-llm-graph-reuse /path/to/CosyVoice3-2512_Q4_K_M.gguf CPU
```

The test fails if a reused graph changes the next-token distribution by more than
the allowed backend rounding tolerance. In particular, it catches the unprotected
`emb_pos` tensor being overwritten after its first use.
