# GPT-OSS Implementation

This repository contains a modular, educational implementation of the
GPT-OSS architecture in PyTorch.

Implemented components:

- Pre-norm RMSNorm transformer blocks
- Grouped-query attention with attention sinks
- Alternating sliding-window and dense attention
- RoPE with YaRN and NTK-by-parts scaling
- Per-layer KV caches for prefill and incremental decoding
- Top-k mixture-of-experts routing
- Clamped SwiGLU
- Safetensors checkpoint loading
- MXFP4 expert-weight dequantization

The model accepts token IDs with shape `(batch, sequence)` and returns FP32
logits with shape `(batch, sequence, vocabulary)`.

```python
import torch

from model import TokenGenerator

generator = TokenGenerator(
    checkpoint="path/to/gpt-oss-checkpoint",
    device=torch.device("cuda"),
)

for token in generator.generate(
    prompt_tokens=[...],
    stop_tokens=[...],
    temperature=0.1,
    max_tokens=100,
):
    print(token)
```

Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```
