import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.config import load_config, hf_auth
from training.training_copy import run_training_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run training copy pipeline")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", "-o", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.output_dir:
        # Update mixed domain training output
        if "training" not in config:
            config["training"] = {}
        config["training"]["output_dir"] = args.output_dir
        
        # Update instruction training output as well
        if "instruction_training" in config:
            config["instruction_training"]["output_dir"] = args.output_dir
        
    hf_token = hf# filepath: c:\Users\luca_\OneDrive\Desktop\Un po' di tutto\EPFL\ML\P02\smiles-mid-training\run_training_copy.py
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.config import load_config, hf_auth
from training.training_copy import run_training_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run training copy pipeline")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", "-o", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.output_dir:
        # Update mixed domain training output
        if "training" not in config:
            config["training"] = {}
        config["training"]["output_dir"] = args.output_dir
        
        # Update instruction training output as well
        if "instruction_training" in config:
            config["instruction_training"]["output_dir"] = args.output_dir
        
    hf_token = hf_auth()
    
    run_training_pipeline(config, hf_token)

if __name__ == "__main__":
    main()