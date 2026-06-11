import torch

from .norm import RMSNorm


def swiglu(
    x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0
) -> torch.Tensor:
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.swiglu_alpha = config.swiglu_alpha

        self.norm = RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size,
            config.num_experts,
            device=device,
            dtype=torch.bfloat16,
        )
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                config.num_experts,
                config.intermediate_size * 2,
                config.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                config.num_experts,
                config.intermediate_size * 2,
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                config.intermediate_size,
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        experts = torch.topk(
            self.gate(t), self.experts_per_token, dim=-1, sorted=True
        )
        expert_weights = torch.softmax(experts.values, dim=-1)
        expert_indices = experts.indices

        mlp1_weight = self.mlp1_weight[expert_indices]
        mlp1_bias = self.mlp1_bias[expert_indices]
        t = torch.einsum("bth,btkih->btki", t, mlp1_weight) + mlp1_bias
        t = swiglu(t, alpha=self.swiglu_alpha, limit=self.swiglu_limit)

        mlp2_weight = self.mlp2_weight[expert_indices]
        mlp2_bias = self.mlp2_bias[expert_indices]
        t = torch.einsum("btki,btkhi->btkh", t, mlp2_weight) + mlp2_bias
        t = torch.einsum("btkh,btk->bth", t, expert_weights)
        return x + t
