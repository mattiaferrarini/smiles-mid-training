import os
import yaml

from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

from .logging import get_logger

LOGGER = get_logger(__name__)

def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
    
def hf_auth(token_override=None):
    load_dotenv()
    token = token_override or os.getenv("HF_TOKEN")

    if not token:
        return None

    try:
        login(token=token, add_to_git_credential=False)
    except Exception as e:
        LOGGER.warning("Hugging Face login failed: %s", e)

    return token
