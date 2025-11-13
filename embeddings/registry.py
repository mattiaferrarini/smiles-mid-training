import torch

REGISTRY = {}

def register_strategy(name):
    name = name.lower().strip()

    def decorator(builder):
        if name in REGISTRY:
            raise ValueError(f"Embedding strategy '{name}' already registered")
        REGISTRY[name] = builder
        return builder

    return decorator


def _ensure_embeddings_available(model):
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError("Model does not expose input embeddings")
    if not hasattr(embeddings, "weight"):
        raise ValueError("Embedding layer does not have accessible weights")
    return embeddings.weight


def _new_section(weight, num_new_tokens):
    if num_new_tokens <= 0:
        raise ValueError("num_new_tokens must be > 0 to initialise new embeddings")
    if num_new_tokens > weight.size(0):
        raise ValueError("Cannot grab new embedding slice larger than embedding matrix")
    return weight[-num_new_tokens:]


@register_strategy("mean_std")
def _build_mean_std(params):
    min_std = float(params.get("min_std", 1e-5))
    std_scale = float(params.get("std_scale", 1.0))

    def apply(model, num_new_tokens):
        if num_new_tokens <= 0:
            return
        weight = _ensure_embeddings_available(model)
        with torch.no_grad():
            existing = weight[:-num_new_tokens]
            new_section = _new_section(weight, num_new_tokens)
            if existing.numel() == 0:
                torch.nn.init.normal_(new_section, mean=0.0, std=0.02)
                return
            mean = existing.mean(dim=0)
            std = existing.std(dim=0).clamp(min=min_std) * std_scale
            sample = torch.randn_like(new_section) * std.unsqueeze(0) + mean.unsqueeze(0)
            new_section.copy_(sample)

    return apply


@register_strategy("average")
def _build_average(params):
    noise_scale = float(params.get("noise_scale", 0.0))
    min_std = float(params.get("min_std", 0.0))

    def apply(model, num_new_tokens):
        if num_new_tokens <= 0:
            return
        weight = _ensure_embeddings_available(model)
        with torch.no_grad():
            existing = weight[:-num_new_tokens]
            new_section = _new_section(weight, num_new_tokens)
            if existing.numel() == 0:
                torch.nn.init.zeros_(new_section)
                return
            mean = existing.mean(dim=0)
            new_section.copy_(mean.unsqueeze(0).expand_as(new_section))
            if noise_scale > 0.0:
                std = existing.std(dim=0).clamp(min=min_std)
                new_section.add_(torch.randn_like(new_section) * std.unsqueeze(0) * noise_scale)

    return apply


@register_strategy("zero")
def _build_zero(params):
    def apply(model, num_new_tokens):
        if num_new_tokens <= 0:
            return
        weight = _ensure_embeddings_available(model)
        with torch.no_grad():
            new_section = _new_section(weight, num_new_tokens)
            new_section.zero_()

    return apply


@register_strategy("normal")
def _build_normal(params):
    mean = float(params.get("mean", 0.0))
    std = float(params.get("std", 0.02))

    def apply(model, num_new_tokens):
        if num_new_tokens <= 0:
            return
        weight = _ensure_embeddings_available(model)
        with torch.no_grad():
            new_section = _new_section(weight, num_new_tokens)
            torch.nn.init.normal_(new_section, mean=mean, std=std)

    return apply


@register_strategy("uniform")
def _build_uniform(params):
    a = float(params.get("a", -0.1))
    b = float(params.get("b", 0.1))

    def apply(model, num_new_tokens):
        if num_new_tokens <= 0:
            return
        weight = _ensure_embeddings_available(model)
        with torch.no_grad():
            new_section = _new_section(weight, num_new_tokens)
            torch.nn.init.uniform_(new_section, a=a, b=b)

    return apply


def build_embedding_strategy(config):
    config = config or {}
    strategy_name = str(config.get("strategy", "mean_std")).lower()
    params = config.get("params", {})

    if strategy_name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown embedding strategy '{strategy_name}'. Available: {available}"
        )
    return REGISTRY[strategy_name](params)


def list_embedding_strategies():
    return sorted(REGISTRY)
