minimal gpt-oss implementation in pytorch.

> [!IMPORTANT]
> this repo is only for learning purposes.

components:

- RMSNorm (pre-norm)
- Grouped Query Attention with attention sinks and sliding window
- RoPE with YaRN
- Mixture-of-Experts (MoE) with gated router
- SwiGLU and KV-cache

reference: these banger repos by [Vizuara](https://github.com/VizuaraAILabs/nano-gpt-oss) and [Hamza Elshafie](https://github.com/HamzaElshafie/gpt-oss-20B)
