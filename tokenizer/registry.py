from transformers import AutoTokenizer
from .hybrid_tokenizer import HybridTokenizer
from .element_tokenizer import ElementTokenizer

TOKENIZERS = {}
TRAINERS = {}

def register_trainer(name):
    normalized = name.lower().strip()

    def decorator(func):
        if normalized in TRAINERS:
            raise ValueError(f"Trainer '{normalized}' already registered")
        TRAINERS[normalized] = func
        return func

    return decorator


def get_trainer(name):
    normalized = name.lower().strip()
    if normalized not in TRAINERS:
        available = ", ".join(sorted(TRAINERS)) or "<none>"
        raise KeyError(f"Trainer '{normalized}' not registered. Available: {available}")
    return TRAINERS[normalized]


def list_trainable_tokenizers():
    return dict(TRAINERS)


def collect_element_symbols():
    element_tokenizer = ElementTokenizer()
    return list(element_tokenizer.get_vocab().keys())


def add_tokens(tokenizer, tokens):
    vocab = tokenizer.get_vocab()
    unique = [tok for tok in tokens if tok not in vocab]
    if unique:
        tokenizer.add_tokens(unique)
    return unique


def build_element_tokenizer(base, params):
    if params.get("include_element_tokens", True):
        add_tokens(base, collect_element_symbols())
    add_tokens(base, params.get("extra_tokens", []))
    return base


def build_hybrid_tokenizer(base, params):
    chem_start = params.get("chem_start", "[CHEM]")
    chem_end = params.get("chem_end", "[/CHEM]")

    chem_tokenizer = ElementTokenizer()
    hybrid = HybridTokenizer(base, chem_tokenizer, chem_start, chem_end)

    if params.get("include_element_tokens", True):
        add_tokens(hybrid.base_tokenizer, collect_element_symbols())
    add_tokens(hybrid.base_tokenizer, params.get("extra_tokens", []))

    if hybrid.pad_token is None and hybrid.base_tokenizer.pad_token is not None:
        hybrid.pad_token = hybrid.base_tokenizer.pad_token
        hybrid.pad_token_id = hybrid.base_tokenizer.pad_token_id
    return hybrid


TOKENIZERS.update(
    {
        "element": build_element_tokenizer,
        "hybrid": build_hybrid_tokenizer,
    }
)


def build_tokenizer_from_config(
    config,
    hf_token,
):
    tokenizer_cfg = config.get("tokenizer") or {}
    tokenizer_type = str(tokenizer_cfg.get("type", "element")).lower()

    model_cfg = config.get("model") or {}
    if "name" not in model_cfg:
        raise ValueError("Model name missing in config file")

    model_name = model_cfg["name"]
    base_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    params = {
        "include_element_tokens": tokenizer_cfg.get("include_element_tokens", True),
        "extra_tokens": tokenizer_cfg.get("extra_tokens", []),
    }

    if tokenizer_type == "hybrid":
        hybrid_cfg = tokenizer_cfg.get("hybrid_params") or {}
        params["chem_start"] = hybrid_cfg.get("chem_start", "[CHEM]")
        params["chem_end"] = hybrid_cfg.get("chem_end", "[/CHEM]")

    if tokenizer_type not in TOKENIZERS:
        available = ", ".join(sorted(TOKENIZERS))
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Available: {available}")

    builder = TOKENIZERS[tokenizer_type]
    return builder(base_tokenizer, params)


def list_registered_tokenizers():
    return sorted(TOKENIZERS.keys())
