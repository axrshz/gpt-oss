import math

import torch

from .cache import Cache
from .norm import RMSNorm
from .rope import RotaryEmbedding


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    sliding_window: int = 0,
    offset: int | torch.Tensor = 0,
) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, n_groups, head_dim = query.shape
    n_ctx = key.shape[1]
    assert key.shape == (batch_size, n_ctx, n_kv_heads, head_dim)
    assert value.shape == (batch_size, n_ctx, n_kv_heads, head_dim)

    if isinstance(offset, torch.Tensor):
        offset = int(offset.item())

    key = key.unsqueeze(3).expand(
        batch_size, n_ctx, n_kv_heads, n_groups, head_dim
    )
    value = value.unsqueeze(3).expand(
        batch_size, n_ctx, n_kv_heads, n_groups, head_dim
    )
    sinks = sinks.reshape(n_kv_heads, n_groups, 1, 1).expand(
        n_kv_heads, n_groups, seq_len, 1
    )

    mask = torch.triu(
        query.new_full((seq_len, n_ctx), -float("inf")), diagonal=offset + 1
    )
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((seq_len, n_ctx), -float("inf")),
            diagonal=offset - sliding_window,
        )

    scores = torch.einsum("bqhmd,bkhmd->bhmqk", query, key)
    scores *= sm_scale
    scores += mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    scores = torch.cat((scores, sinks.unsqueeze(0)), dim=-1)
    weights = torch.softmax(scores, dim=-1)[..., :-1]
    output = torch.einsum("bhmqk,bkhmd->bqhmd", weights, value)
    return output.reshape(batch_size, seq_len, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_groups = self.num_attention_heads // self.num_key_value_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        self.sinks = torch.nn.Parameter(
            torch.empty(
                config.num_attention_heads, device=device, dtype=torch.bfloat16
            )
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            max_context_length=(
                config.initial_context_length * int(config.rope_scaling_factor)
            ),
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(
        self, x: torch.Tensor, cache: Cache | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(self.norm(x))

        q_end = self.num_attention_heads * self.head_dim
        k_end = q_end + self.num_key_value_heads * self.head_dim
        q = qkv[:, :, :q_end].contiguous()
        k = qkv[:, :, q_end:k_end].contiguous()
        v = qkv[:, :, k_end:].contiguous()

        q = q.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        )
        k = k.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        v = v.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        offset = (
            cache.offset.clone()
            if cache is not None
            else torch.zeros((1,), dtype=torch.long, device=x.device)
        )
        q, k = self.rope(q, k, offset)
        if cache is not None:
            k, v = cache.extend(k, v)

        q = q.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.num_groups,
            self.head_dim,
        )
        output = sdpa(
            q,
            k,
            v,
            self.sinks,
            self.sm_scale,
            self.sliding_window,
            offset,
        )
        return x + self.out(output)
