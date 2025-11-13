import json
import yaml
import typer

from pathlib import Path
from transformers import AutoModelForCausalLM

from utils.config import hf_auth, load_config
from .utils import resolve_embedding_config, summarise_embedding_sections

from utils.logging import get_logger
from tokenizer.registry import build_tokenizer_from_config
from embeddings.registry import build_embedding_strategy, list_embedding_strategies

LOGGER = get_logger(__name__)


def list_embeddings():
    typer.echo("Registered embedding strategies:")
    for name in list_embedding_strategies():
        typer.echo(f"  - {name}")


def _write_embedding_metadata(
    output_dir,
    strategy_name,
    model_name,
    num_tokens,
    stats,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "embeddings": {
            "strategy": strategy_name,
            "num_new_tokens": num_tokens,
            "artifact_path": str(output_dir),
            "model": model_name,
        }
    }

    metadata_path = output_dir / "embedding_metadata.yaml"
    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    stats_payload = {
        "strategy": strategy_name,
        "model": model_name,
        "num_new_tokens": num_tokens,
        "stats": stats,
    }
    stats_path = output_dir / "embedding_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2)

    LOGGER.info(
        "Wrote embedding evaluation artifacts to %s (metadata) and %s (stats)",
        metadata_path,
        stats_path,
    )


def evaluate_embedding_strategy(
    config_path,
    strategy_override,
    num_tokens,
    report_path,
    output_dir,
    hf_token=None,
    model_override=None,
):
    if isinstance(num_tokens, str):
        num_tokens = int(num_tokens)
    if num_tokens <= 0:
        typer.echo("ERROR: num_tokens must be a positive integer.")
        raise typer.Exit(code=1)

    config_dict = load_config(config_path)
    embedding_cfg = resolve_embedding_config(config_dict.get("embeddings"), strategy_override)
    token = hf_auth(token_override=hf_token)

    if model_override:
        model_cfg = dict(config_dict.get("model") or {})
        model_cfg["name"] = model_override
        config_dict["model"] = model_cfg

    model_name = config_dict["model"]["name"]
    LOGGER.info(
        "Evaluating embedding strategy '%s' using model %s", embedding_cfg["strategy"], model_name
    )

    tokenizer = build_tokenizer_from_config(config_dict, token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    original_vocab_size = model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(original_vocab_size + num_tokens)

    strategy_fn = build_embedding_strategy(embedding_cfg)
    strategy_fn(model, num_tokens)

    weight = model.get_input_embeddings().weight.detach()
    stats = summarise_embedding_sections(weight, num_tokens)

    typer.echo(
        f"Embedding stats after applying strategy '{embedding_cfg['strategy']}' (num_new_tokens={num_tokens}):"
    )
    for key, value in stats.items():
        typer.echo(f"  {key}: {value:.6f}")

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **stats,
                    "strategy": embedding_cfg["strategy"],
                    "num_new_tokens": num_tokens,
                    "model": model_name,
                },
                handle,
                indent=2,
            )
        LOGGER.info("Wrote embedding stats to %s", report_path)
        typer.echo(f"INFO: Embedding stats written to {report_path}")

    if output_dir is not None:
        _write_embedding_metadata(
            output_dir,
            embedding_cfg["strategy"],
            model_name,
            num_tokens,
            stats,
        )
        typer.echo(f"INFO: Embedding metadata written to {output_dir}")
