import yaml
import typer
import logging

from pathlib import Path

from utils.logging import setup_logging
from utils.config import hf_auth, load_config
from training.training import run_training_pipeline
from training.benchmarking import evaluate_baselines
from training.baselines import download_baseline_models
from tokenizer.registry import list_trainable_tokenizers
from embeddings.cli import evaluate_embedding_strategy, list_embeddings
from tokenizer.cli import evaluate_tokenizer, list_tokenizers, preview_tokenizer, train_tokenizer

TRAINABLE_TOKENIZER_SCHEMES = sorted(list(list_trainable_tokenizers().keys()))
DEFAULT_TOKENIZER_SCHEME = (
    TRAINABLE_TOKENIZER_SCHEMES[0] if TRAINABLE_TOKENIZER_SCHEMES else "bpe"
)

app = typer.Typer(help="Test novel tokenisation schemes and mid-stage training strategies for open-source chemical LLMs")

@app.callback()
def configure_logging(
    log_level=typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    ),
):
    level = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level=level)

@app.command("tokenizer-list")
def tokenizer_list_command():
    list_tokenizers()

@app.command("tokenizer-preview")
def tokenizer_preview_command(
    config=typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Config file used to instantiate the tokenizer",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    text=typer.Argument(..., help="Text snippet to tokenize"),
    tokenizer_type=typer.Option(
        None,
        "--tokenizer-type",
        "--scheme",
        "-s",
        help="Override the tokenizer type defined in the config.",
    ),
    model_name=typer.Option(
        None,
        "--model",
        "-m",
        help="Override the base model name defined in the config.",
    ),
    hf_token=typer.Option(
        None,
        "--hf-token",
        help="Optional Hugging Face token override for gated models.",
    ),
):
    preview_tokenizer(config, text, tokenizer_type, model_name, hf_token)

@app.command("tokenizer-evaluate")
def tokenizer_evaluate_command(
    config=typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Config used to instantiate the tokenizer.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    text_path=typer.Option(
        ...,
        "--text",
        help="Path to a newline-delimited text file for evaluation.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    limit=typer.Option(
        1000,
        help="Maximum number of lines to sample from the evaluation corpus.",
    ),
    report_path=typer.Option(
        None,
        "--report",
        help="Optional path to write the metrics as JSON.",
    ),
    tokenizer_type=typer.Option(
        None,
        "--tokenizer-type",
        "--scheme",
        "-s",
        help="Override the tokenizer type defined in the config.",
    ),
    model_name=typer.Option(
        None,
        "--model",
        "-m",
        help="Override the base model name defined in the config.",
    ),
    hf_token=typer.Option(
        None,
        "--hf-token",
        help="Optional Hugging Face token override for gated models.",
    ),
):
    evaluate_tokenizer(
        config,
        text_path,
        limit,
        report_path,
        tokenizer_type,
        model_name,
        hf_token,
    )


@app.command("tokenizer-train")
def tokenizer_train_command(
    tokenizer_type=typer.Option(
        DEFAULT_TOKENIZER_SCHEME,
        "--tokenizer-type",
        "--scheme",
        "-s",
        help=(
            "Tokenization scheme to train. Available: "
            + ", ".join(TRAINABLE_TOKENIZER_SCHEMES or ["bpe"])
        ),
    ),
    dataset=typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to a newline-delimited text corpus used for tokenizer training.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir=typer.Option(
        Path("artifacts/tokenizers/bpe"),
        "--output-dir",
        "-o",
        help="Directory where the trained tokenizer will be saved.",
    ),
    vocab_size=typer.Option(32000, help="Vocabulary size for the trained tokenizer."),
    min_frequency=typer.Option(2, help="Minimum token frequency during training."),
    limit=typer.Option(
        None,
        help="Optional maximum number of lines to read from the dataset when training.",
    ),
    hf_token=typer.Option(
        None,
        "--hf-token",
        help="Optional Hugging Face token override for gated models.",
    ),
):
    train_tokenizer(
        tokenizer_type,
        dataset,
        output_dir,
        vocab_size,
        min_frequency,
        limit,
        hf_token=hf_token,
    )

@app.command("embedding-list")
def embedding_list_command():
    list_embeddings()


@app.command("embedding-evaluate")
def embedding_evaluate_command(
    config=typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    strategy=typer.Option(
        None,
        "--strategy",
        "-s",
        help="Override the embedding initialisation strategy defined in the config.",
    ),
    num_tokens=typer.Option(
        64,
        "--num-tokens",
        "-n",
        help="Number of freshly added tokens to initialise for evaluation.",
    ),
    report_path=typer.Option(
        None,
        "--report",
        help="Optional path to write embedding summary statistics as JSON.",
    ),
    output_dir=typer.Option(
        Path("artifacts/embeddings/evaluations"),
        "--output-dir",
        "-o",
        help="Directory to store embedding evaluation metadata and stats.",
    ),
    model_name=typer.Option(
        None,
        "--model",
        "-m",
        help="Override the base model name defined in the config.",
    ),
    hf_token=typer.Option(
        None,
        "--hf-token",
        help="Optional Hugging Face token override for gated models.",
    ),
):
    evaluate_embedding_strategy(
        config,
        strategy,
        num_tokens,
        report_path,
        output_dir,
        hf_token,
        model_name,
    )

@app.command("get-baseline")
def get_baseline_command(
    config=typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help=(
            "Path to YAML configuration file; defaults to config.model.name when"
            " no --model overrides are provided."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_names=typer.Option(
        None,
        "--model",
        "-m",
        help="Override the default model name; can be provided multiple times.",
    ),
    output_dir=typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Root directory where baseline artifacts will be stored.",
    ),
    hf_token=typer.Option(
        None,
        "--hf-token",
        help="Optional Hugging Face token override for gated models.",
    ),
):
    config_dict = load_config(config)
    token = hf_auth(token_override=hf_token)

    download_baseline_models(
        config=config_dict,
        hf_token=token,
        output_dir=output_dir,
        only_models=model_names,
    )



@app.command("train")
def train_command(
    config=typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_name=typer.Option(
        None,
        "--model-name",
        "-m",
        help="Override the model name defined in the config.",
    ),
    data_folder=typer.Option(
        None,
        "--data-folder",
        "-d",
        help="Override the data folder defined in the config.",
    ),
    output_dir=typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override the output directory defined in the config.",
    ),
):
    with open(config, "r") as fh:
        cfg = yaml.safe_load(fh)

    if model_name:
        cfg.setdefault("model", {})["name"] = model_name
    if data_folder:
        cfg.setdefault("data", {})["data_folder"] = str(data_folder)
    if output_dir:
        cfg.setdefault("training", {})["output_dir"] = str(output_dir)

    token = hf_auth()
    run_training_pipeline(cfg, token, embedding_override=None)

@app.command("benchmark-baselines")
def benchmark_baselines_command(
    baselines_root=typer.Option(
        Path("artifacts/baselines"),
        "--baselines-root",
        "-b",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing downloaded baseline checkpoints.",
    ),
    model_names=typer.Option(
        None,
        "--model",
        "-m",
        help="Optional subset of baseline model folders to evaluate (repeatable).",
    ),
    output_dir=typer.Option(
        Path("artifacts/benchmark-results"),
        "--output-dir",
        "-o",
        help="Destination for ChemBench/ChemPile/ChemIQ reports.",
    ),
    prompt_type=typer.Option(
        "instruction",
        "--prompt-type",
        help="ChemBench prompt template to use.",
    ),
    topics=typer.Option(
        None,
        "--topic",
        "-t",
        help="Limit ChemBench to specific topics (repeatable).",
    ),
    device=typer.Option(
        "auto",
        "--device",
        help="Device override passed to the HF model wrapper.",
    ),
    torch_dtype=typer.Option(
        "auto",
        "--torch-dtype",
        help="Torch dtype string (e.g. float16, bfloat16) or 'auto'.",
    ),
    max_new_tokens=typer.Option(
        512,
        "--max-new-tokens",
        help="Maximum tokens generated per ChemBench prompt.",
    ),
    temperature=typer.Option(
        0.0,
        "--temperature",
        help="Sampling temperature; >0 enables sampling.",
    ),
    chempile_command=typer.Option(
        None,
        "--chempile-command",
        help="Shell template for invoking the official ChemPile runner (expects {model_dir}/{output_dir}).",
    ),
    chemiq_command=typer.Option(
        None,
        "--chemiq-command",
        help="Shell template for invoking the official ChemIQ runner (expects {model_dir}/{output_dir}).",
    ),
    config=typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file used to resolve default model if not provided.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    if not model_names:
        config_dict = load_config(config)
        model_cfg = config_dict.get("model")
        if isinstance(model_cfg, str):
            model_names = [model_cfg]
        elif isinstance(model_cfg, dict) and "name" in model_cfg:
            model_names = [model_cfg["name"]]

    summaries = evaluate_baselines(
        baselines_root=baselines_root,
        model_filters=list(model_names) if model_names else None,
        output_root=output_dir,
        prompt_type=prompt_type,
        topics=list(topics) if topics else None,
        device=device,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        chempile_command=chempile_command,
        chemiq_command=chemiq_command,
    )
    typer.echo(f"Wrote summaries for {len(summaries)} model(s) to {output_dir}")


if __name__ == "__main__":
    app()
