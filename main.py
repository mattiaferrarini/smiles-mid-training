
import typer
import logging

from pathlib import Path

from utils.logging import setup_logging
from utils.config import hf_auth, load_config
from training.training import run_training_pipeline
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
    embedding_strategy=typer.Option(
        None,
        "--embedding-strategy",
        "-es",
        help="Override the embedding initialisation strategy defined in the config.",
    ),
    hf_token=typer.Option(
        None,
        "--hf-token",
        help="Optional Hugging Face token override for gated models.",
    ),
):
    config_dict = load_config(config)
    token = hf_auth(token_override=hf_token)
    run_training_pipeline(config_dict, token, embedding_override=embedding_strategy)


if __name__ == "__main__":
    app()
