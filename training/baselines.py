import os
from pathlib import Path
from dotenv import load_dotenv
from utils.logging import get_logger
from huggingface_hub import snapshot_download

LOGGER = get_logger(__name__)


def download_baseline_artifacts(config, output_path):
    """
    Downloads the baseline model artifacts from HuggingFace based on the provided configuration

    Args:
        config (dict): Configuration dictionary containing model details
        output_path (str): Path to save the downloaded artifacts
    """
    load_dotenv()

    model_cfg = config.get("model")
    if model_cfg is None:
        raise Exception("Config is missing 'model' section")

    model_name = model_cfg.get("name")
    revision = model_cfg.get("revision")

    if not model_name:
        raise Exception("Config section 'model' is missing 'name' field")

    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path) / model_name.replace("/", "-")

    LOGGER.info(
        "Ensuring baseline model %s is available at %s", model_name, output_path
    )

    snapshot_download(
        repo_id=model_name,
        revision=revision,
        local_dir=output_path,
        local_dir_use_symlinks=False,
    )

    LOGGER.info("Verified baseline model artifacts for %s", model_name)
