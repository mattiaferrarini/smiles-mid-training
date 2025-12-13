import yaml
from pathlib import Path
import os
from huggingface_hub import login
from dotenv import load_dotenv  #
from utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def hf_auth():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    if token:
        try:
            login(token=token, add_to_git_credential=False)
            LOGGER.info("Hugging Face authentication successful (using HF_TOKEN)")
        except Exception as e:
            LOGGER.warning(
                f"Hugging Face authentication failed (invalid token): {e}",
                exc_info=True,
            )
    else:
        try:
            login(add_to_git_credential=False)
            LOGGER.info(
                "HF_TOKEN not found; using local cached Hugging Face authentication if available"
            )
        except Exception:
            LOGGER.warning(
                "No Hugging Face authentication found; proceeding as anonymous user",
                exc_info=True,
            )
