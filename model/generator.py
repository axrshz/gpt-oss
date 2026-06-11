from collections.abc import Generator

import torch

from .cache import Cache
from .transformer import Transformer


class TokenGenerator:
    def __init__(
        self, checkpoint: str, device: str | torch.device = "cuda"
    ) -> None:
        self.device = torch.device(device)
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

    def _create_caches(
        self, batch_size: int, cache_size: int
    ) -> list[Cache]:
        config = self.model.config
        return [
            Cache(
                batch_size=batch_size,
                n_ctx=cache_size,
                n_kv_heads=config.num_key_value_heads,
                d_head=config.head_dim,
                device=self.device,
            )
            for _ in range(config.num_hidden_layers)
        ]

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
    ) -> Generator[int | tuple[int, float], None, None]:
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        config = self.model.config
        max_context = (
            config.initial_context_length * int(config.rope_scaling_factor)
        )
        available_tokens = max_context - len(prompt_tokens)
        if available_tokens <= 0:
            raise ValueError(
                f"Prompt length {len(prompt_tokens)} reaches or exceeds "
                f"the maximum context length {max_context}"
            )
        generation_limit = (
            min(max_tokens, available_tokens) if max_tokens > 0 else available_tokens
        )
        cache_size = len(prompt_tokens) + generation_limit
        caches = self._create_caches(batch_size=1, cache_size=cache_size)

        input_tensor = torch.as_tensor(
            [prompt_tokens], dtype=torch.long, device=self.device
        )
        logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)

        for token_index in range(generation_limit):
            if token_index > 0:
                input_tensor = torch.as_tensor(
                    [[predicted_token]], dtype=torch.long, device=self.device
                )
                logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)

            if temperature == 0.0:
                predicted_token = int(torch.argmax(logits, dim=-1).item())
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                predicted_token = int(
                    torch.multinomial(probabilities, num_samples=1).item()
                )

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                yield predicted_token, float(logprobs[predicted_token].item())
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break
