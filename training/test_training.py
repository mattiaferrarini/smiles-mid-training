import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to Python path to enable imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_config
from custom_tokenizers.assemble_tokenizer import assemble_tokenizer

if __name__ == "__main__":
    load_dotenv()

    # Load configuration
    config_path = "configs/default_trl.yaml"
    config = load_config(config_path)
    config["tokenizer"]["type"] = "character"

    # Assemble tokenizer based on configuration
    tokenizer = assemble_tokenizer(config)