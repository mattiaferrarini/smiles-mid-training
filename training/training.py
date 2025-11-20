import os
import json
import shutil
import argparse
import logging
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)
from datasets import load_dataset
from dotenv import load_dotenv

from utils.config import load_config, hf_auth
from utils.logging import get_logger

# Constants
DDP_BACKEND = "nccl"
DATA_LOADER_NUM_WORKERS = 4

LOGGER = get_logger(__name__)

def get_default_training_config():
    return {
        "output_dir": "artifacts/runs/default",
        "epochs": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "warmup_steps": 100,
        "save_steps": 1000,
        "logging_steps": 50,
        "bf16": True,
        "fp16": False,
        "weight_decay": 0.0,
        "evaluation_strategy": "no",
        "save_strategy": "no",
        "seed": 42,
        "gradient_checkpointing": False,
    }


def normalize_training_config(raw_cfg):
    cfg = {**get_default_training_config(), **raw_cfg}
    return cfg


def detect_training_mode(config):
    tokenizer_cfg = config.get("tokenizer", {})
    embeddings_cfg = config.get("embeddings", {})
    
    # Advanced mode if custom tokenizer or embeddings specified
    has_custom_tokenizer = tokenizer_cfg.get("type") in ["element", "hybrid"]
    has_custom_embeddings = embeddings_cfg.get("strategy") is not None
    
    if has_custom_tokenizer or has_custom_embeddings:
        return "advanced"
    return "baseline"


def validate_config(config, mode):
    if "model" not in config or "name" not in config["model"]:
        raise ValueError("Config must contain model.name")
    
    if "data" not in config:
        raise ValueError("Config must contain data section")
    
    if mode == "advanced":
        if "tokenizer" in config and config["tokenizer"].get("type") == "hybrid":
            # Validate hybrid tokenizer params if specified
            hybrid_params = config["tokenizer"].get("hybrid_params", {})
            if not hybrid_params.get("chem_start") or not hybrid_params.get("chem_end"):
                LOGGER.warning("Hybrid tokenizer without chem_start/chem_end markers")
    
    LOGGER.info(f"Configuration validated for {mode} mode")

def run_jobreport(output_dir, stage = "start"):
    jr_path = shutil.which("jobreport") or os.path.join(os.getcwd(), "jobreport")
    if not jr_path or not os.path.exists(jr_path):
        logging.warning("jobreport binary not found on PATH or repo root; skipping jobreport.")
        return None

    os.makedirs(os.path.join(output_dir, "jobreport"), exist_ok=True)
    jobid = os.getenv("SLURM_JOB_ID", f"local-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_path = os.path.join(output_dir, "jobreport", f"{jobid}-{stage}.json")

    cmd = [jr_path, "--json"]
    try:
        logging.info(f"Running jobreport: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode != 0:
            logging.warning(f"jobreport exited with code {res.returncode}; stderr: {res.stderr.strip()}")
            # still write whatever we got
        with open(out_path, "w") as fh:
            fh.write(res.stdout)
        try:
            data = json.loads(res.stdout) if res.stdout.strip() else None
        except Exception:
            logging.warning("Could not parse jobreport JSON output; saved raw output to %s", out_path)
            data = None
        return data
    except FileNotFoundError:
        logging.warning("jobreport binary not found (FileNotFoundError), skipping jobreport.")
        return None


def analyze_jobreport(data):
    summary = {
        "found": False,
        "total_gpus": 0,
        "gpu_devices": [],
        "nodes": None,
        "gpus_per_node": None,
        "gpu_usage": {},
    }
    if not data:
        return summary

    # Helper: recursive search for keys/values
    def _walk(obj, path=[]):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield (path + [k], k, v)
                yield from _walk(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield (path + [str(i)], str(i), v)
                yield from _walk(v, path + [str(i)])

    # Look for gpu-related entries
    gpu_names = []
    gpu_count = 0
    nodes = None
    usages = {}
    for p, k, v in _walk(data):
        key_lower = str(k).lower()
        if "node" in key_lower and isinstance(v, (int, str)) and nodes is None:
            try:
                nodes = int(v)
            except Exception:
                pass
        if "gpu" in key_lower or "accelerator" in key_lower or "device" in key_lower:
            # If value is list-like containing device names
            if isinstance(v, list):
                # collect strings
                for item in v:
                    if isinstance(item, str):
                        gpu_names.append(item)
                gpu_count += len(v)
            elif isinstance(v, str):
                # device name or description
                if any(tok.upper() in v.upper() for tok in ["A100", "V100", "RTX", "H100", "Tesla"]):
                    gpu_names.append(v)
                    gpu_count += 1
            elif isinstance(v, (int, float)) and ("count" in key_lower or "total" in key_lower or key_lower.endswith("s")):
                try:
                    gpu_count = max(gpu_count, int(v))
                except Exception:
                    pass
        # look for usage percentages / utilization fields
        if key_lower in ("utilization", "util") or key_lower.endswith("util") or "util" in key_lower:
            # if value is numeric or contains percentage info, record it
            if isinstance(v, (int, float)):
                usages['/'.join(p)] = float(v)
            elif isinstance(v, str):
                # try to extract a number from strings like '45%'
                try:
                    s = v.strip().rstrip('%')
                    usages['/'.join(p)] = float(s)
                except Exception:
                    pass
    
    # attach any found usages into summary
    if usages:
        summary['gpu_usage'] = usages
    
    # Fallbacks
    if gpu_count == 0 and gpu_names:
        gpu_count = len(gpu_names)

    summary["found"] = gpu_count > 0 or bool(gpu_names)
    summary["total_gpus"] = gpu_count
    summary["gpu_devices"] = gpu_names
    summary["nodes"] = nodes
    if nodes and gpu_count:
        try:
            summary["gpus_per_node"] = int(gpu_count / nodes)
        except Exception:
            summary["gpus_per_node"] = None
    return summary

def prepare_tokenizer_baseline(model_name, hf_token):
    LOGGER.info(f"Loading baseline tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_tokenizer_advanced(config, hf_token):
    from tokenizer.registry import build_tokenizer_from_config
    
    LOGGER.info("Loading advanced tokenizer with custom configuration")
    tokenizer = build_tokenizer_from_config(config, hf_token)
    return tokenizer


def prepare_model(model_name, hf_token):
    LOGGER.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype="auto",
        device_map=None  # Let accelerator handle device mapping
    )
    return model

def resize_embeddings_if_needed(model, tokenizer, config):
    from embeddings.registry import build_embedding_strategy
    from embeddings.utils import summarise_embedding_sections
    
    original_vocab_size = model.get_input_embeddings().weight.size(0)
    target_vocab_size = len(tokenizer)
    
    if target_vocab_size <= original_vocab_size:
        LOGGER.info("No embedding resize required (vocab unchanged)")
        return
    
    # Resize embeddings
    model.resize_token_embeddings(target_vocab_size)
    num_new_tokens = target_vocab_size - original_vocab_size
    
    # Apply embedding initialization strategy
    embeddings_cfg = config.get("embeddings", {})
    strategy_name = embeddings_cfg.get("strategy", "mean_std")
    strategy_params = embeddings_cfg.get("params", {})
    
    LOGGER.info(f"Initializing {num_new_tokens} new embeddings with strategy: {strategy_name}")
    strategy = build_embedding_strategy({
        "strategy": strategy_name,
        **strategy_params
    })
    strategy(model, num_new_tokens)
    
    # Log statistics
    weight = model.get_input_embeddings().weight.detach()
    stats = summarise_embedding_sections(weight, num_new_tokens)
    LOGGER.info(
        f"Resized embeddings from {original_vocab_size} to {target_vocab_size}"
    )
    LOGGER.info(
        "New embedding stats: " + 
        ", ".join(f"{k}={v:.6f}" for k, v in stats.items() if k.startswith("new_"))
    )


def initialize_model_and_tokenizer(config, mode, hf_token):
    model_name = config["model"]["name"]
    
    # Load tokenizer
    if mode == "baseline":
        tokenizer = prepare_tokenizer_baseline(model_name, hf_token)
    else:
        tokenizer = prepare_tokenizer_advanced(config, hf_token)
    
    # Load model
    model = prepare_model(model_name, hf_token)
    
    # Set pad token in model config
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply gradient checkpointing if requested
    training_cfg = config.get("training", {})
    if training_cfg.get("gradient_checkpointing", False):
        LOGGER.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Resize embeddings if in advanced mode
    if mode == "advanced":
        resize_embeddings_if_needed(model, tokenizer, config)
    
    return model, tokenizer

def load_training_dataset(config, tokenizer):
    data_cfg = config.get("data", {})
    data_folder = data_cfg.get("data_folder", "data/processed")
    text_field = data_cfg.get("text_field", "text")
    
    LOGGER.info(f"Loading dataset from {data_folder}")
    dataset = load_dataset("arrow", data_dir=data_folder, data_files="**/*.arrow")
    dataset = dataset["train"]
    
    LOGGER.info(f"Tokenizing text field: {text_field}")
    dataset = dataset.map(lambda x: tokenizer(x[text_field]), batched=True)
    
    LOGGER.info(f"Dataset loaded: {len(dataset)} samples")
    return dataset

def train(
    config,
    model,
    tokenizer,
    output_dir,
    use_wandb,
    wandb_run_name,
    wandb_config = None
):
    # Extract and normalize training config
    training_cfg = normalize_training_config(config.get("training", {}))
    
    # Load and process dataset
    dataset = load_training_dataset(config, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    report_to = "wandb" if use_wandb else "none"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(training_cfg["epochs"]),
        warmup_steps=int(training_cfg["warmup_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        save_steps=int(training_cfg["save_steps"]),
        logging_steps=int(training_cfg["logging_steps"]),
        bf16=bool(training_cfg["bf16"]),
        fp16=bool(training_cfg["fp16"]),
        evaluation_strategy=training_cfg["evaluation_strategy"],
        save_strategy=training_cfg["save_strategy"],
        seed=int(training_cfg["seed"]),
        remove_unused_columns=False,
        dataloader_num_workers=DATA_LOADER_NUM_WORKERS,
        ddp_backend=DDP_BACKEND,
        report_to=report_to,
    )
    
    LOGGER.info(f"Training for {training_cfg['epochs']} epochs with batch size {training_cfg['batch_size']}")
    LOGGER.info(f"Learning rate: {training_cfg['learning_rate']}, warmup steps: {training_cfg['warmup_steps']}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Record W&B run info if enabled
    if use_wandb:
        try:
            import wandb
            wandb_run_id = getattr(wandb.run, "id", None)
            if wandb_run_id is not None:
                with open(os.path.join(output_dir, "wandb_run.json"), "w") as fh:
                    json.dump({
                        "wandb_run_id": wandb_run_id,
                        "run_name": wandb_run_name
                    }, fh, indent=2)
                LOGGER.info(f"W&B run ID saved: {wandb_run_id}")
        except Exception as e:
            LOGGER.warning(f"Could not save W&B run info: {e}")
    
    return trainer

def setup_wandb(config, output_dir, timestamp):
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        return False, None, None
    
    try:
        import wandb
        
        # Login to W&B
        try:
            wandb.login(key=wandb_api_key, timeout=30)
        except TypeError:
            # Older wandb versions may not accept timeout kwarg
            wandb.login(key=wandb_api_key)
        
        # Prepare run metadata
        model_name = config["model"]["name"]
        training_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        
        wandb_run_name = f"{Path(output_dir).name}-{model_name.replace('/', '-')}-{timestamp}"
        wandb_config = {
            "model_name": model_name,
            "data_folder": data_cfg.get("data_folder", "data/processed"),
            "text_field": data_cfg.get("text_field", "text"),
            "per_device_train_batch_size": training_cfg.get("batch_size", 2),
            "num_train_epochs": training_cfg.get("epochs", 2),
            "learning_rate": training_cfg.get("learning_rate", 2e-5),
        }
        
        wandb.init(
            project="smiles-mid-training",
            name=wandb_run_name,
            config=wandb_config
        )
        
        LOGGER.info(f"W&B initialized: {wandb_run_name}")
        return True, wandb_run_name, wandb_config
        
    except Exception as e:
        LOGGER.warning(f"Could not initialize W&B: {e}")
        return False, None, None

def run_training_pipeline(config, hf_token, embedding_override = None):
    # Load environment variables
    load_dotenv()
    
    # Setup SLURM environment for distributed training
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator()
    
    # Detect training mode
    mode = detect_training_mode(config)
    validate_config(config, mode)
    
    if accelerator.is_main_process:
        LOGGER.info("=" * 80)
        LOGGER.info("UNIFIED TRAINING PIPELINE")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Mode: {mode.upper()}")
        LOGGER.info(f"Model: {config['model']['name']}")
        LOGGER.info(f"Data: {config.get('data', {}).get('data_folder', 'N/A')}")
        LOGGER.info(f"Distributed: {accelerator.distributed_type}")
        LOGGER.info(f"Process: {accelerator.process_index}/{accelerator.num_processes}")
        LOGGER.info("=" * 80)
    
    # Create timestamped output directory
    training_cfg = config.get("training", {})
    base_output_dir = training_cfg.get("output_dir", "artifacts/runs/default")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = str(Path(base_output_dir) / timestamp)
    
    if accelerator.is_main_process:
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Output directory: {final_output_dir}")
        
        # Save config for reproducibility
        with open(Path(final_output_dir) / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Setup W&B (optional)
    use_wandb, wandb_run_name, wandb_config = setup_wandb(config, base_output_dir, timestamp)
    
    # Run jobreport at start (best-effort)
    if accelerator.is_main_process:
        jr_data_start = run_jobreport(final_output_dir, stage="start")
        jr_summary_start = analyze_jobreport(jr_data_start)
        LOGGER.info(f"Jobreport start summary: {jr_summary_start}")
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(config, mode, hf_token)
    
    # Training
    trainer = None
    try:
        trainer = train(
            config=config,
            model=model,
            tokenizer=tokenizer,
            output_dir=final_output_dir,
            use_wandb=use_wandb,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config
        )
    finally:
        # Run jobreport at end (best-effort)
        if accelerator.is_main_process:
            jr_data_end = run_jobreport(final_output_dir, stage="end")
            jr_summary_end = analyze_jobreport(jr_data_end)
            LOGGER.info(f"Jobreport end summary: {jr_summary_end}")
    
    # Save final model
    if accelerator.is_main_process and trainer is not None:
        final_model_dir = str(Path(final_output_dir) / "final-model")
        LOGGER.info(f"Saving final model to {final_model_dir}")
        trainer.save_model(final_model_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(final_model_dir)
        LOGGER.info("Training complete!")
    
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
