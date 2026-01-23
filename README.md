gpt oss implementation in pytorch.

components:

- RoPE with YaRN
- Mixture-of-Experts (MoE) with gated router
- Grouped Query Attention with attention sinks and sliding window
- SwiGLU, RMSNorm (pre-norm), KV-cache
