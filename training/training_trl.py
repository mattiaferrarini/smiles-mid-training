import os
import sys
from pathlib import Path
import yaml

# Add parent directory to Python path to enable imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import transformers
import transformers.utils.import_utils
import transformers.trainer

def patched_check_safety():
    """
    Completely bypass the torch.load security check.
    Required because checking optimizer.pt (legacy pickle) causes a crash 
    in newer transformers versions.
    """
    return

transformers.utils.import_utils.check_torch_load_is_safe = patched_check_safety
transformers.trainer.check_torch_load_is_safe = patched_check_safety

from accelerate import Accelerator, InitProcessGroupKwargs
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

from datasets import load_dataset, interleave_datasets
import datasets
from dotenv import load_dotenv
import wandb
import torch
import torch.distributed as dist
import time
from utils.logging import get_logger
import argparse
from utils.config import load_config
from datetime import timedelta
import json
from custom_tokenizers.assemble_tokenizer import assemble_tokenizer
from embeddings.embedding_initializer import initialize_embeddings as init_embeddings_fn


LOGGER = get_logger(__name__)

START_SMILES, END_SMILES = "[START_SMILES]", "[END_SMILES]"

class MultiGPUResourcesCallback(TrainerCallback):
    def __init__(self, log_steps):
        super().__init__()
        self.log_steps = log_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Print memory usage for every rank to ensure load balancing
        if torch.cuda.is_available():
            rank = int(os.environ.get("RANK", 0))
            if state.global_step % self.log_steps == 0: 
                current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"[Step {state.global_step}] Rank {rank}: {current_mem:.2f} GB (Max: {max_mem:.2f} GB)")


def prepare_training_args(config, output_dir):
    strategy = config["distributed"]["strategy"]

    # Common arguments
    args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["training"]["per_device_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "num_train_epochs": config["training"]["epochs"],
        "warmup_ratio": config["training"]["warmup_ratio"],
        "lr_scheduler_type": config["training"]["lr_scheduler_type"],
        "learning_rate": config["training"]["learning_rate"],
        "bf16": config["training"]["bf16"],
        "fp16": config["training"]["fp16"],
        "remove_unused_columns": False,
        "dataloader_num_workers": config["training"]["num_workers"],
        "report_to": config["training"]["report_to"],
        "gradient_checkpointing": config["training"]["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": config["training"]["save_strategy"],
        "save_steps": config["training"]["save_steps"],
        "logging_steps": config["training"]["logging_steps"],
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": config["training"]["prefetch_factor"],
        "packing": True,
        "accelerator_config": {
            "dispatch_batches": False,
        },
        "dataloader_drop_last": True,
        # --- FIX CRITICO QUI SOTTO: max_seq_length ---
        "max_length": config["training"]["max_length"],
        "include_tokens_per_second": True,
        "include_num_input_tokens_seen": True,
    }
    if strategy == "fsdp":
        # FSDP-specific arguments
        fsdp_conf = config["distributed"]["fsdp"]
        args_dict["fsdp"] = fsdp_conf["policy"]
        fsdp_inner_config = fsdp_conf["config"].copy()

        if "fsdp_transformer_layer_cls_to_wrap" in fsdp_inner_config:
            args_dict["fsdp_transformer_layer_cls_to_wrap"] = fsdp_inner_config.pop("fsdp_transformer_layer_cls_to_wrap")
            LOGGER.info(f"FSDP Wrapping Layer: {args_dict['fsdp_transformer_layer_cls_to_wrap']}")
        
        if args_dict["gradient_checkpointing"]:
            fsdp_inner_config["activation_checkpointing"] = True
        args_dict["gradient_checkpointing"] = False       
 
        args_dict["fsdp_config"] = fsdp_inner_config
        
        LOGGER.info(f"Training Strategy: FSDP ({args_dict['fsdp']})")
    elif strategy == "ddp":
        # DDP-specific arguments
        ddp_conf = config["distributed"]["ddp"]
        args_dict["ddp_backend"] = ddp_conf["backend"]
        args_dict["ddp_find_unused_parameters"] = ddp_conf["find_unused_parameters"]
        LOGGER.info(f"Training Strategy: DDP (Backend: {args_dict['ddp_backend']})")
    else:
        raise ValueError(f"Unsupported distributed strategy: {strategy}")
    
    return SFTConfig(**args_dict)


def build_tokenizer(config):
    # Use utility function to assemble tokenizer
    return assemble_tokenizer(config)


def initialize_embeddings(model, tokenizer, config):
    initialization_strategy = config["tokenizer"].get("embedding_initialization", "default")
    
    # DEBUG: Verifica che tipo di tokenizer abbiamo
    print(f"[DEBUG] initialize_embeddings called with tokenizer type: {type(tokenizer)}")
    if hasattr(tokenizer, "chem_tokenizer"):
        print("[DEBUG] Tokenizer is Hybrid (chem_tokenizer found).")
    else:
        print("[DEBUG] Tokenizer appears to be standard (NO chem_tokenizer found).")

    if initialization_strategy == "random":
        print("Initializing embeddings with strategy: random")
        model.resize_token_embeddings(len(tokenizer))
    else:
        print(f"Initializing embeddings with strategy: {initialization_strategy}")
        # Chiamata alla funzione esterna
        model = init_embeddings_fn(model, tokenizer, strategy=initialization_strategy)
        
    return model


def prepare_dataset(tokenizer, config, accelerator):
    # Load and process main dataset
    dataset = load_dataset(
        "arrow", 
        data_dir=config["data"]["data_folder"], 
        data_files=config["data"]["data_files_pattern"], 
        split="train",
        # streaming=True
    ).select_columns([config["data"]["text_field"]])

    # Rename text column if necessary
    my_text_col = config["data"]["text_field"]
    if my_text_col != "text":
        dataset = dataset.rename_column(my_text_col, "text")
    dataset = dataset.select_columns(["text"])

    # Select a portion of the dataset if specified
    print(f"Original dataset size: {(dataset.data.nbytes / (1024 ** 3)):.2f} GB")
    portion_of_data_used = config["data"]["portion_of_data_used"]
    if portion_of_data_used < 1.0:
        total_samples = dataset.num_rows
        samples_to_use = int(total_samples * portion_of_data_used)
        dataset = dataset.select(range(samples_to_use))
    print(f"Dataset size after selecting {portion_of_data_used*100}%: {(dataset.data.nbytes / (1024 ** 3)):.2f} GB")

    # Determine number of fineweb chunks to use
    SIZE_OF_CHUNK_GB = 2.15
    probabilities = config["data_mix"]["probabilities"]

    dataset_gb = dataset.data.nbytes / (1024 ** 3)
    num_chunks = round(dataset_gb / SIZE_OF_CHUNK_GB * probabilities[0] / probabilities[1])
    subset_path = config["data_mix"]["external_subset_name"]
    print(f"Using {num_chunks} chunks from FineWeb based on dataset size of {dataset_gb:.2f} GB")

    fineweb_data_files = []
    FILES_PER_MAIN_INDEX = config["data_mix"]["files_per_main_index"]

    for i in range(num_chunks):
        main_index = i // FILES_PER_MAIN_INDEX  
        sub_index = i % FILES_PER_MAIN_INDEX   
        main_part = f"{main_index:03d}"  
        sub_part = f"{sub_index:05d}"    
        file_name = f"{subset_path}/{main_part}_{sub_part}.parquet"
        fineweb_data_files.append(file_name)

    fineweb = load_dataset(
        config["data_mix"]["external_dataset_name"],
        split="train",
        data_files=fineweb_data_files,
        # streaming=True
    ).select_columns(["text"])
    
    # dataset = dataset.select(range(100000))
    # fineweb = fineweb.select(range(100000))

    # Tokenize the text field

    # Silence progress bars on non-main processes
    if not accelerator.is_main_process:
        datasets.utils.logging.disable_progress_bar()

    def tokenize_and_split(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["training"]["max_length"],
            return_overflowing_tokens=True, # Split long samples
            stride=config["training"]["stride_size"], # Overlap between chunks
            padding=False
        )
        
        if "overflow_to_sample_mapping" in outputs:
            outputs.pop("overflow_to_sample_mapping")
            
        return outputs
    
    # Process dataset with main process
    with accelerator.main_process_first():
        # Tokenize datasets 
        fineweb = fineweb.map(
            tokenize_and_split,
            batched=True,
            num_proc=config["training"]["preprocessing_num_workers"],
            batch_size=config["training"]["preprocessing_batch_size"],
            remove_columns=fineweb.column_names, 
            load_from_cache_file=True
        )

        dataset = dataset.map(
            tokenize_and_split,
            batched=True,
            num_proc=config["training"]["preprocessing_num_workers"],
            batch_size=config["training"]["preprocessing_batch_size"],
            remove_columns=dataset.column_names, 
            load_from_cache_file=True
        )

        print("Tokenized samples in dataset:", len(dataset))
        print("Tokenized samples in fineweb:", len(fineweb))
        print("Total tokenized samples:", len(dataset) + len(fineweb))

    # Re-enable logging
    if not accelerator.is_main_process:
        datasets.utils.logging.enable_progress_bar()   

    combined_dataset = interleave_datasets(
        [fineweb, dataset],
        probabilities=probabilities,
        seed=config["training"]["seed"], 
        stopping_strategy="first_exhausted"
    )
    return combined_dataset


def get_last_checkpoint(output_dir):
    # Check for existing checkpoints
    checkpoint_dir = None
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by checkpoint number and get the latest
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(output_dir, checkpoints[-1])
            LOGGER.info(f"Found existing checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def train(config, accelerator, output_dir):
    # Check for existing checkpoints
    checkpoint_dir = get_last_checkpoint(output_dir)
    
    # Load tokenizer   
    tokenizer = build_tokenizer(config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None # Let accelerator handle device mapping
    )
   
    # Explicitly sync model config with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    
    # Initialize new tokens' embeddings
    model = initialize_embeddings(model, tokenizer, config)
    
    # Enable gradient checkpointing
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # Prepare dataset
    combined_dataset = prepare_dataset(tokenizer, config, accelerator)
    
    # Training arguments
    training_args = prepare_training_args(config, output_dir)
    resource_logging_steps = config["training"]["resource_logging_steps"]    

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        processing_class=tokenizer, # pass tokenizer here
        callbacks=[
            MultiGPUResourcesCallback(resource_logging_steps),
        ]
    )    

    # Train the model (resume from checkpoint if available)
    if checkpoint_dir:
        print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("Starting training from scratch")
        trainer.train()
    return trainer


def init_wandb(config):
    # Login wandb
    wandb.login()

    # Initialize wandb run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "smiles"),
        name=config["job"]["name"],
        config={
            "model": config["model"]["name"],
            "lr": config["training"]["learning_rate"],
            "batch_size": config["training"]["per_device_batch_size"],
            "epochs": config["training"]["epochs"],
            "data_folder": config["data"]["data_folder"],
            "text_field": config["data"]["text_field"],
            "num_workers": config["training"]["num_workers"],
        },
        # group=os.getenv("WANDB_GROUP", "DDP"),
    )
    return run


def train_model(config, output_dir):
    load_dotenv()

    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # Initialize distributed training
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[timeout])

    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Save config to output directory in both YAML and JSON formats
        config_output_path_yaml = os.path.join(output_dir, "training_config.yaml")
        with open(config_output_path_yaml, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved config to {config_output_path_yaml}")

        config_output_path_json = os.path.join(output_dir, "training_config.json")
        with open(config_output_path_json, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Saved config to {config_output_path_json}")
        
        print("Starting fine-tuning.")
        print(f"Distributed: {accelerator.distributed_type}")
        print(f"Process: {accelerator.process_index}/{accelerator.num_processes}")
        print(f"Config: {config}")
        print(f"Output dir: {output_dir}")
    
    # Initialize wandb
    if config["training"]["report_to"] == "wandb":
        run = init_wandb(config)

    # Start training
    trainer = train(config, accelerator, output_dir)
    LOGGER.info(f"[RANK {int(os.environ.get('RANK', 0))}] Finished training")
    
    # Save final model
    final_model_dir = f"{output_dir}/final_model" 
    trainer.save_model(final_model_dir)

    if accelerator.is_main_process:
        print(f"Saved model to {final_model_dir}")
    
    if config["training"]["report_to"] == "wandb":
        run.finish()
    
    # Ensure group termination    
    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune Gemma model.")
    parser.add_argument("--config-path", "-c", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)
    output_dir = args.output_dir
    train_model(config, output_dir)

    
