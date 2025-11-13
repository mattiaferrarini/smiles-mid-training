import json
import yaml
import typer
from pathlib import Path

from utils.logging import get_logger
from utils.config import hf_auth, load_config

from tokenizer.registry import (
    build_tokenizer_from_config,
    get_trainer,
    list_trainable_tokenizers,
    list_registered_tokenizers,
)
from tokenizer.utils import compute_tokenizer_metrics, dump_metrics, iter_text_samples

LOGGER = get_logger(__name__)


def _load_config_if_available(config_path):
    if config_path is None:
        return {}
    config = load_config(config_path)
    return config or {}


def _resolve_tokenizer_config(
    config_path,
    tokenizer_type,
    model_name,
):
    config_dict = dict(_load_config_if_available(config_path))

    model_cfg = dict(config_dict.get("model") or {})
    if model_name:
        model_cfg["name"] = model_name
    if "name" not in model_cfg:
        raise typer.BadParameter("Model name must be provided via config or --model")
    config_dict["model"] = model_cfg

    tokenizer_cfg = dict(config_dict.get("tokenizer") or {})
    if tokenizer_type:
        tokenizer_cfg["type"] = tokenizer_type.lower()
    tokenizer_cfg.setdefault("type", "element")
    config_dict["tokenizer"] = tokenizer_cfg

    return config_dict


def _resolve_hf_token(token_override):
    return hf_auth(token_override=token_override)


def _write_training_metadata(
    output_dir,
    tokenizer_type,
    tokenizer_vocab_size,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tokenizer": {
            "type": tokenizer_type,
            "artifact_path": str(output_dir),
        }
    }
    if tokenizer_vocab_size is not None:
        metadata["tokenizer"]["vocab_size"] = tokenizer_vocab_size

    metadata_path = output_dir / "tokenizer_metadata.yaml"
    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    snippet = {
        "tokenizer": {
            "type": tokenizer_type,
            "params": {
                "pretrained_path": str(output_dir),
            },
        }
    }
    snippet_path = output_dir / "tokenizer_config.json"
    with snippet_path.open("w", encoding="utf-8") as handle:
        json.dump(snippet, handle, indent=2)

    LOGGER.info(
        "Wrote tokenizer metadata files: %s (metadata), %s (config snippet)",
        metadata_path,
        snippet_path,
    )

def list_tokenizers():
    typer.echo("Registered tokenizers:")
    trainable = set(list_trainable_tokenizers())
    for name in list_registered_tokenizers():
        suffix = " (trainable)" if name in trainable else ""
        typer.echo(f"  - {name}{suffix}")

def preview_tokenizer(
    config_path,
    text,
    tokenizer_type=None,
    model_name=None,
    hf_token=None,
):
    config_dict = _resolve_tokenizer_config(config_path, tokenizer_type, model_name)
    token = _resolve_hf_token(hf_token)

    tokenizer = build_tokenizer_from_config(config_dict, token)
    tokens = tokenizer.tokenize(text)

    LOGGER.info(
        "Previewed tokenizer '%s' with %d tokens", config_dict['tokenizer']['type'], len(tokens)
    )
    typer.echo(f"Tokens ({len(tokens)}): {tokens}")

def evaluate_tokenizer(
    config_path,
    text_path,
    limit,
    report_path,
    tokenizer_type=None,
    model_name=None,
    hf_token=None,
):
    config_dict = _resolve_tokenizer_config(config_path, tokenizer_type, model_name)
    token = _resolve_hf_token(hf_token)
    tokenizer = build_tokenizer_from_config(config_dict, token)

    LOGGER.info(
        "Evaluating tokenizer '%s' on %s", config_dict['tokenizer']['type'], text_path
    )
    typer.echo(
        f"INFO: Evaluating tokenizer type '{config_dict['tokenizer']['type']}' on text file {text_path}"
    )

    corpus = list(iter_text_samples(text_path, limit=limit))
    if not corpus:
        typer.echo("ERROR: Evaluation corpus is empty.")
        raise typer.Exit(code=1)

    metrics = compute_tokenizer_metrics(tokenizer, corpus)

    typer.echo("Tokenizer metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            typer.echo(f"  {key}: {value:.4f}")
        else:
            typer.echo(f"  {key}: {value}")

    dump_metrics(metrics, report_path)

def train_tokenizer(
    tokenizer_type,
    dataset,
    output_dir,
    vocab_size,
    min_frequency,
    limit,
    hf_token=None,
):
    tokenizer_type = tokenizer_type.lower()
    trainer = get_trainer(tokenizer_type)

    typer.echo(f"INFO: Training tokenizer '{tokenizer_type}' from dataset {dataset}")
    LOGGER.info(
        "Training tokenizer '%s' (vocab_size=%d, min_frequency=%d) from %s",
        tokenizer_type,
        vocab_size,
        min_frequency,
        dataset,
    )

    _resolve_hf_token(hf_token)

    tokenizer = trainer(
        dataset=dataset,
        output_dir=output_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        limit=limit,
    )

    vocab_size_value = None
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab_size_value = len(tokenizer.get_vocab())
        except Exception:
            vocab_size_value = None

    _write_training_metadata(output_dir, tokenizer_type, vocab_size_value)

    typer.echo(f"INFO: Saved tokenizer '{tokenizer_type}' artifacts to {output_dir}")
