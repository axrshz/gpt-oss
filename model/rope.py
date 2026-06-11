import math

import torch


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        max_context_length: int = 131072,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.max_context_length = max_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

        cos, sin = self._compute_cos_sin(max_context_length, device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _compute_concentration_and_inv_freq(
        self, device: torch.device | None
    ) -> tuple[float, torch.Tensor]:
        pair_indices = torch.arange(
            0, self.head_dim, 2, dtype=torch.float32, device=device
        )
        frequencies = self.base ** (pair_indices / self.head_dim)

        if self.scaling_factor <= 1.0:
            return 1.0, 1.0 / frequencies

        concentration = 0.1 * math.log(self.scaling_factor) + 1.0
        d_half = self.head_dim // 2
        low = (
            d_half
            * math.log(
                self.initial_context_length / (self.ntk_beta * 2 * math.pi)
            )
            / math.log(self.base)
        )
        high = (
            d_half
            * math.log(
                self.initial_context_length / (self.ntk_alpha * 2 * math.pi)
            )
            / math.log(self.base)
        )
        assert 0 < low < high < d_half - 1

        interpolation = 1.0 / (self.scaling_factor * frequencies)
        extrapolation = 1.0 / frequencies
        ramp = (
            torch.arange(d_half, dtype=torch.float32, device=device) - low
        ) / (high - low)
        mask = 1 - ramp.clamp(0, 1)
        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        return concentration, inv_freq

    def _compute_cos_sin(
        self, num_tokens: int, device: torch.device | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concentration, inv_freq = self._compute_concentration_and_inv_freq(device)
        positions = torch.arange(num_tokens, dtype=torch.float32, device=device)
        frequencies = torch.einsum("i,j->ij", positions, inv_freq)
        return (
            frequencies.cos() * concentration,
            frequencies.sin() * concentration,
        )

    @staticmethod
    def _rotate(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        cos = cos.unsqueeze(0).unsqueeze(2).to(x.dtype)
        sin = sin.unsqueeze(0).unsqueeze(2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[1]
        indices = (
            torch.arange(num_tokens, device=query.device, dtype=torch.long) + offset
        )
        indices = indices % self.max_context_length

        cos = self.cos.index_select(0, indices)
        sin = self.sin.index_select(0, indices)
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)
