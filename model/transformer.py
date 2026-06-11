import json
import os

import torch

from .attention import AttentionBlock
from .cache import Cache
from .config import ModelConfig
from .mlp import MLPBlock
from .norm import RMSNorm
from .weights import Checkpoint


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, layer_idx, device)

    def forward(
        self, x: torch.Tensor, cache: Cache | None = None
    ) -> torch.Tensor:
        x = self.attn(x, cache=cache)
        return self.mlp(x)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.configs = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.block = torch.nn.ModuleList(
            TransformerBlock(config, layer_idx, device)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(
        self, x: torch.Tensor, caches: list[Cache] | None = None
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected token IDs with shape (batch, sequence), got {x.shape}"
            )
        if caches is None:
            caches = [None] * len(self.block)
        elif len(caches) != len(self.block):
            raise ValueError(
                f"Expected {len(self.block)} caches, got {len(caches)}"
            )

        x = self.embedding(x)
        for block, cache in zip(self.block, caches):
            x = block(x, cache=cache)
        return self.unembedding(self.norm(x)).float()

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        device = torch.device(device)
        config_path = os.path.join(path, "config.json")
        with open(config_path, encoding="utf-8") as config_file:
            config = ModelConfig(**json.load(config_file))

        model = Transformer(config, device=device)
        model.eval()
        checkpoint = Checkpoint(path, device, config.num_hidden_layers)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                tensor = checkpoint.get(name)
                if tensor.shape != parameter.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {name}: checkpoint has "
                        f"{tensor.shape}, model expects {parameter.shape}"
                    )
                parameter.copy_(tensor)
        return model
