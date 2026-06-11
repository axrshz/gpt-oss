import torch

from model import Cache, ModelConfig, Transformer


def tiny_config() -> ModelConfig:
    return ModelConfig(
        num_hidden_layers=2,
        num_experts=4,
        experts_per_token=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        sliding_window=4,
        initial_context_length=16,
        rope_scaling_factor=1.0,
    )


def initialize_parameters(model: Transformer) -> None:
    torch.manual_seed(7)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.05, 0.05)


def test_forward_shape_and_dtype() -> None:
    config = tiny_config()
    model = Transformer(config)
    initialize_parameters(model)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model(tokens)

    assert logits.shape == (2, 3, config.vocab_size)
    assert logits.dtype == torch.float32


def test_cached_decode_matches_full_sequence() -> None:
    config = tiny_config()
    model = Transformer(config).eval()
    initialize_parameters(model)
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    full_logits = model(tokens)
    caches = [
        Cache(
            batch_size=1,
            n_ctx=tokens.shape[1],
            n_kv_heads=config.num_key_value_heads,
            d_head=config.head_dim,
        )
        for _ in range(config.num_hidden_layers)
    ]
    cached_logits = torch.cat(
        [
            model(tokens[:, token_index : token_index + 1], caches=caches)
            for token_index in range(tokens.shape[1])
        ],
        dim=1,
    )

    torch.testing.assert_close(cached_logits, full_logits, rtol=2e-2, atol=2e-2)
