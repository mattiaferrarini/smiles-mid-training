import typer
import logging
from pathlib import Path
from utils.config import load_config
from utils.logging import setup_logging
from evaluate.benchmark import run_chemiq, run_chembench
from training.baselines import download_baseline_artifacts
from custom_tokenizers.build_tokenizer import build_tokenizer
from instructions.instruction import run_instruction_tuning
from evaluate.likelihood_eval import run_likelihood_eval
from training.training_trl import train_model

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

@app.command("download-baseline")
def download_baseline_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to configuration file", exists=True, file_okay=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output-path", "-o", help="Where to save the downloaded artifacts", file_okay=False, dir_okay=True),
):
    config_dict = load_config(config)
    download_baseline_artifacts(config_dict, output_path)

@app.command("build-tokenizer")
def build_tokenizer_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config file", exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory where to save the tokenizer", file_okay=False, dir_okay=True),
):
    build_tokenizer(output_dir=str(output_dir), config_path=str(config))

@app.command("train-model")
def train_model_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to configuration file", exists=True, file_okay=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output-path", "-o", help="Where to save results", file_okay=False, dir_okay=True),
):
    config_dict = load_config(config)
    train_model(config_dict, str(output_path))

@app.command("instruction-tuning")
def train_instruction_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to configuration file", exists=True, file_okay=True, dir_okay=False),
    model_path: str = typer.Option(None, "--model-path", "-m", help="Optional: override model name/path from config (can be local path or HF Hub ID)"),
):
    run_instruction_tuning(config_path=str(config), model_path=model_path)

@app.command("likelihood-eval")
def run_likelihood_eval_command(
    model_path: str = typer.Option(..., "--model-path", "-m", help="Path to the model or checkpoints directory to benchmark"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory where to save results", file_okay=False, dir_okay=True),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
    annotate_smiles: bool = typer.Option(False, "--annotate-smiles", help="Annotate SMILES with [START_SMILES] tags"),
):
    run_likelihood_eval(model_path=model_path, output_dir=str(output_dir), debug=debug, annotate_smiles=annotate_smiles)

@app.command("run-chembench")
def run_chembench_command(
    model_path: Path = typer.Option(..., "--model-path", "-m", help="Path to the model to benchmark", exists=True, file_okay=False, dir_okay=True),
    output_path: Path = typer.Option(..., "--output-path", "-o", help="Where to save results", file_okay=False, dir_okay=True),
):
    run_chembench(model_path, output_path)

@app.command("run-chemiq")
def run_chemiq_command(
    model_path: Path = typer.Option(..., "--model-path", "-m", help="Path to the model to benchmark", exists=True, file_okay=False, dir_okay=True),
    chemiq_path: Path = typer.Option(..., "--chemiq-path", help="Path to the ChemIQ repository", exists=True, file_okay=False, dir_okay=True),
    output_path: Path = typer.Option(..., "--output-path", "-o", help="Where to save results", file_okay=False, dir_okay=True),
):
    run_chemiq(model_path, chemiq_path, output_path)

if __name__ == "__main__":
    app()