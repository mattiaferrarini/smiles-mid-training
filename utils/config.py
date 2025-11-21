import yaml
from pathlib import Path

def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)