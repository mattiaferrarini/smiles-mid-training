import os
import yaml

from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
from typing import Any, Dict, Optional

from .logging import get_logger

LOGGER = get_logger(__name__)

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
    
def hf_auth() -> Optional[str]:
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    if token:
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            LOGGER.warning("Hugging Face login failed: %s", e)
    else:
        LOGGER.warning("No Hugging Face token found")

    return token
