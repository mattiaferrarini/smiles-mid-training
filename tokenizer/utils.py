
import json
from pathlib import Path
from utils.logging import get_logger

LOGGER = get_logger(__name__)

def iter_text_samples(path, limit=None):
    path = Path(path)
    if isinstance(limit, str):
        limit = int(limit)
    with path.open("r", encoding="utf-8") as src:
        for idx, line in enumerate(src):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                yield line


def tokens_for_text(tokenizer, text):
    try:
        tokens = tokenizer.tokenize(text)
        if tokens:
            return list(tokens)
    except Exception:
        pass

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoding.get("input_ids", [])
    return tokenizer.convert_ids_to_tokens(token_ids)


def compute_tokenizer_metrics(
    tokenizer,
    texts,
):
    total_tokens = 0
    total_words = 0
    continuation_tokens = 0

    continuation_prefix = getattr(tokenizer, "continuing_subword_prefix", None)
    if continuation_prefix is None and hasattr(tokenizer, "backend_tokenizer"):
        continuation_prefix = getattr(
            getattr(tokenizer.backend_tokenizer, "model", None),
            "continuing_subword_prefix",
            None,
        )

    sentencepiece_marker = "▁"

    for text in texts:
        words = text.split()
        tokens = tokens_for_text(tokenizer, text)

        total_words += len(words)
        total_tokens += len(tokens)

        if continuation_prefix:
            continuation_tokens += sum(token.startswith(continuation_prefix) for token in tokens)
        else:
            continuation_tokens += sum(not token.startswith(sentencepiece_marker) for token in tokens)

    metrics = {
        "fertility": total_tokens / total_words if total_words else 0.0,
        "continuation_rate": continuation_tokens / total_tokens if total_tokens else 0.0,
        "total_tokens": float(total_tokens),
        "total_words": float(total_words),
    }
    return metrics


def dump_metrics(metrics, report_path):
    if report_path is None:
        return
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("Written tokenizer metrics report to %s", report_path)
