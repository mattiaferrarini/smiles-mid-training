import typer
import logging
import tempfile
from pathlib import Path
from utils.config import load_config
from utils.logging import setup_logging
from evaluate.benchmark import run_chemiq, run_chembench
from training.baselines import download_baseline_artifacts
from custom_tokenizers.build_tokenizer import build_tokenizer
from instructions.instruction import run_instruction_tuning
from evaluate.likelihood_eval import run_likelihood_eval
from evaluate.fertility import evaluate_tokenizers_fertility
from evaluate.embeddings import evaluate_embeddings
from training.training_trl import train_model

from custom_tokenizers import (
    CharacterTokenizer,
    ElementTokenizer,
    ElementAllParenthesisTokenizer,
    ElementAromaticsTokenizer,
    ElementNoParenthesisTokenizer,
    ElementRingsTokenizer,
    HybridTokenizer,
    APETokenizer,
    APEHFTokenizer,
    APEWPHFTokenizer,
    ChemAPETokenizer,
)

TOKENIZERS = {
    "CharacterTokenizer": CharacterTokenizer,
    "ElementTokenizer": ElementTokenizer,
    "ElementAllParenthesisTokenizer": ElementAllParenthesisTokenizer,
    "ElementAromaticsTokenizer": ElementAromaticsTokenizer,
    "ElementNoParenthesisTokenizer": ElementNoParenthesisTokenizer,
    "ElementRingsTokenizer": ElementRingsTokenizer,
    "HybridTokenizer": HybridTokenizer,
    "APETokenizer": APETokenizer,
    "APEHFTokenizer": APEHFTokenizer,
    "APEWPHFTokenizer": APEWPHFTokenizer,
    "ChemAPETokenizer": ChemAPETokenizer,
}

app = typer.Typer(
    help="Test novel tokenisation schemes and mid-stage training strategies for open-source chemical LLMs"
)


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


@app.command("download-baseline")
def download_baseline_command(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(
        ...,
        "--output-path",
        "-o",
        help="Where to save the downloaded artifacts",
        file_okay=False,
        dir_okay=True,
    ),
):
    config_dict = load_config(config)
    download_baseline_artifacts(config_dict, output_path)


@app.command("build-tokenizer")
def build_tokenizer_command(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory where to save the tokenizer",
        file_okay=False,
        dir_okay=True,
    ),
):
    build_tokenizer(output_dir=str(output_dir), config_path=str(config))


@app.command("train-model")
def train_model_command(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(
        ...,
        "--output-path",
        "-o",
        help="Where to save results",
        file_okay=False,
        dir_okay=True,
    ),
):
    config_dict = load_config(config)
    train_model(config_dict, str(output_path))


@app.command("instruction-tuning")
def train_instruction_command(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_path: str = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Optional: override model name/path from config (can be local path or HF Hub ID)",
    ),
):
    run_instruction_tuning(config_path=str(config), model_path=model_path)


@app.command("likelihood-eval")
def run_likelihood_eval_command(
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to the model or checkpoints directory to benchmark",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory where to save results",
        file_okay=False,
        dir_okay=True,
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
    annotate_smiles: bool = typer.Option(
        False, "--annotate-smiles", help="Annotate SMILES with [START_SMILES] tags"
    ),
):
    run_likelihood_eval(
        model_path=model_path,
        output_dir=str(output_dir),
        debug=debug,
        annotate_smiles=annotate_smiles,
    )


@app.command("run-chembench")
def run_chembench_command(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to the model to benchmark",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_path: Path = typer.Option(
        ...,
        "--output-path",
        "-o",
        help="Where to save results",
        file_okay=False,
        dir_okay=True,
    ),
):
    run_chembench(model_path, output_path)


@app.command("run-chemiq")
def run_chemiq_command(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to the model to benchmark",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    chemiq_path: Path = typer.Option(
        ...,
        "--chemiq-path",
        help="Path to the ChemIQ repository",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_path: Path = typer.Option(
        ...,
        "--output-path",
        "-o",
        help="Where to save results",
        file_okay=False,
        dir_okay=True,
    ),
):
    run_chemiq(model_path, chemiq_path, output_path)


@app.command("fertility-eval")
def run_fertility_eval_command(
    registry_path: Path = typer.Option(
        ...,
        "--registry-path",
        "-r",
        help="Path to registry of tokenizer configurations",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    tokenizers_folder: Path = typer.Option(
        ...,
        "--tokenizers-folder",
        "-t",
        help="Directory containing the tokenizers to evaluate",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    dataset_path: Path = typer.Option(
        ...,
        "--dataset-path",
        "-d",
        help="Path to dataset containing SMILES",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_folder: Path = typer.Option(
        ...,
        "--output-folder",
        "-o",
        help="Where to save fertility evaluation results",
        file_okay=False,
        dir_okay=True,
    ),
):
    evaluate_tokenizers_fertility(str(registry_path), str(tokenizers_folder), str(dataset_path), str(output_folder))


@app.command("evaluate-embeddings")
def evaluate_embeddings_command(
    checkpoint_folder: str = typer.Option(
        ...,
        "--checkpoint-folder",
        "-c",
        help="Path to the model checkpoint folder",
    ),
    dataset_path: str = typer.Option(
        ...,
        "--dataset-path",
        "-d",
        help="Path to the dataset CSV file",
    ),
    smiles_col: str = typer.Option(
        ...,
        "--smiles-col",
        "-s",
        help="Name of the column containing SMILES strings",
    ),
    label_col: str = typer.Option(
        ...,
        "--label-col",
        "-l",
        help="Name of the column containing labels",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "-o",
        help="Path to save the embeddings plot",
    ),
):
    evaluate_embeddings(
        checkpoint_folder,
        dataset_path,
        smiles_col,
        label_col,
        output_path,
    )


@app.command("test-tokenizer")
def test_tokenizer_command(
    tokenizer_name: str = typer.Option(
        ..., "--tokenizer", "-t", help="Name of the tokenizer class"
    ),
    text: str = typer.Option(..., "--text", "-x", help="Text to tokenize"),
    vocab_path: Path = typer.Option(
        None,
        "--vocab-path",
        "-v",
        help="Path to vocabulary file (optional)",
        exists=True,
        dir_okay=False,
    ),
):

    if tokenizer_name not in TOKENIZERS:
        typer.echo(
            f"Tokenizer {tokenizer_name} not found. Available: {list(TOKENIZERS.keys())}"
        )
        raise typer.Exit(code=1)

    TokenizerClass = TOKENIZERS[tokenizer_name]
    if vocab_path:
        tokenizer = TokenizerClass(vocab_file=str(vocab_path))
    else:
        tokenizer = TokenizerClass()
        print("Warning: No vocabulary file provided. Using default/empty vocabulary.")

    s = text
    print(f"\nTRAINING ON TEXT: '{s}'")
    tokenizer.create_vocabulary(s)

    print("\nORIGINAL VOCABULARY CONTENT")
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

    print(f"Total items: {len(vocab)}")
    print(f"Reported len(tokenizer): {len(tokenizer)}\n")
    print(f"{'ID':<5} | {'Token':<10}\n")
    for token, id in sorted_vocab:
        print(f"{id:<5} | {token:<10}")

    # Encoding Test
    encoded = tokenizer.encode(s)
    print(f"\nEncoded IDs: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded:     {decoded}")

    # Save & Load Cycle
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"\nSaving tokenizer to {tmpdirname}...")
        tokenizer.save_pretrained(tmpdirname)

        print("Loading tokenizer from saved files...")
        loaded_tk = TokenizerClass.from_pretrained(tmpdirname)

        print("\nLOADED VOCABULARY CONTENT")
        loaded_vocab = loaded_tk.get_vocab()

        if vocab == loaded_vocab:
            print("Dictionary content is identical")
        else:
            print("Mismatch found in dictionary content")
            print(f"Original keys: {len(vocab)}, Loaded keys: {len(loaded_vocab)}")

        ids_original = tokenizer.encode(s)
        ids_loaded = loaded_tk.encode(s)

        if ids_original == ids_loaded:
            print("Same IDs")
        else:
            print("Failed, they don't have same ids")
            print(f"Original IDs: {ids_original}")
            print(f"Loaded IDs:   {ids_loaded}")

        print("\nSUCCESS: Process completed.")


if __name__ == "__main__":
    app()
