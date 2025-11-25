import typer
import logging
from pathlib import Path
from utils.config import load_config
from utils.logging import setup_logging
from training.training import train_model
from training.benchmark import run_chemiq, run_chembench
from training.baselines import download_baseline_artifacts

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

@app.command("train-model")
def train_model_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to configuration file", exists=True, file_okay=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output-path", "-o", help="Where to save results", file_okay=False, dir_okay=True),
):
    config_dict = load_config(config)
    train_model(config_dict, output_path)

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
