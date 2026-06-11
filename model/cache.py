import torch


class Cache:
    def __init__(
        self,
        batch_size: int,
        n_ctx: int,
        n_kv_heads: int,
        d_head: int,
        device: torch.device | None = None,
    ) -> None:
        shape = (batch_size, n_ctx, n_kv_heads, d_head)
        self.k = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.v = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self) -> None:
        self.k.zero_()
        self.v.zero_()
        self.offset.zero_()

    def repeat_interleave(self, repeats: int) -> None:
        self.k = self.k.repeat_interleave(repeats, dim=0)
        self.v = self.v.repeat_interleave(repeats, dim=0)

    def extend(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_new = k.shape[1]
        start = int(self.offset.item())
        end = start + n_new
        if end > self.k.shape[1]:
            raise ValueError(
                f"KV cache capacity exceeded: need {end} positions, "
                f"capacity is {self.k.shape[1]}"
            )

        indices = torch.arange(start, end, device=k.device, dtype=torch.long)
        self.k.index_copy_(1, indices, k)
        self.v.index_copy_(1, indices, v)
        self.offset.add_(n_new)
        return self.k, self.v
