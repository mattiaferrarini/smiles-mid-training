import torch

def resolve_embedding_config(
    base_config,
    strategy_override=None,
):
    config = dict(base_config or {})
    if strategy_override is not None:
        config["strategy"] = strategy_override
    config.setdefault("strategy", "mean_std")
    params = config.get("params")
    if params is None:
        config["params"] = {}
    elif not isinstance(params, dict):
        raise ValueError("'embeddings.params' must be a mapping when provided")
    return config


def summarise_embedding_sections(
    weight_matrix,
    num_new_tokens,
):
    if num_new_tokens <= 0:
        raise ValueError("num_new_tokens must be > 0 for summary")

    existing = weight_matrix[:-num_new_tokens]
    new_section = weight_matrix[-num_new_tokens:]

    with torch.no_grad():
        stats = {
            "existing_mean": existing.mean().item(),
            "existing_std": existing.std(unbiased=False).item(),
            "existing_norm": existing.norm(p=2).item(),
            "new_mean": new_section.mean().item(),
            "new_std": new_section.std(unbiased=False).item(),
            "new_norm": new_section.norm(p=2).item(),
        }
    return stats
