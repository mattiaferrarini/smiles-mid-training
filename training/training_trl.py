import os
import sys
from pathlib import Path

# Add parent directory to Python path to enable imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

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


LOGGER = get_logger(__name__)

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


class ThroughputCallback(TrainerCallback):
    def __init__(self, log_steps, max_seq_length, accelerator):
        super().__init__()
        self.log_steps = log_steps
        self.max_seq_length = max_seq_length
        self.accelerator = accelerator
        self.last_time = None
        self.last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Only print on the main process to avoid clutter
        if state.global_step % self.log_steps == 0 and self.accelerator.is_main_process:
            current_time = time.time()
            
            # Skip the very first log (step 0 or start) as we have no delta
            if self.last_time is not None:
                time_delta = current_time - self.last_time
                steps_delta = state.global_step - self.last_step
                
                if time_delta > 0:
                    # Calculate total samples processed across all GPUs
                    # Batch size per device * Gradient Accumulation * World Size
                    batch_size_total = (
                        args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
                    )
                    
                    # Since packing=True, every sample is filled to max_seq_length
                    total_tokens = batch_size_total * self.max_seq_length * steps_delta
                    
                    tokens_per_sec = total_tokens / time_delta
                    
                    LOGGER.info(f"[Step {state.global_step}] "
                                f"Time: {time_delta:.2f}s | "
                                f"Throughput: {tokens_per_sec:.2f} tokens/sec")

            self.last_time = current_time
            self.last_step = state.global_step


def prepare_training_args(config, output_dir):
    strategy = config["distributed"]["strategy"]

    # Common arguments
    args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["training"]["per_device_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        # "num_train_epochs": config["training"]["epochs"],
        "max_steps": 50000,
        #"warmup_steps": config["training"]["warmup_steps"],
        "warmup_ratio": config["training"]["warmup_ratio"],
        "lr_scheduler_type": config["training"]["lr_scheduler_type"],
        "learning_rate": config["training"]["learning_rate"],
        "bf16": config["training"]["bf16"],
        "fp16": config["training"]["fp16"],
        "remove_unused_columns": False,
        "dataloader_num_workers": 4,  # config["training"]["num_workers"],
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
        "max_length": config["training"]["max_length"],
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


def train(config, accelerator, output_dir):
    # Load tokenizer   
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
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

    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # Load and process dataset
    subset_path = config["data_mix"]["external_subset_name"]
    num_chunks = config["data_mix"]["num_chunks_to_use"]
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
        streaming=True
    ).select_columns(["text"])

    dataset = load_dataset(
        "arrow", 
        data_dir=config["data"]["data_folder"], 
        data_files=config["data"]["data_files_pattern"], 
        split="train",
        streaming=True
    ).select_columns([config["data"]["text_field"]])

    my_text_col = config["data"]["text_field"]
    if my_text_col != "text":
        dataset = dataset.rename_column(my_text_col, "text")
    dataset = dataset.select_columns(["text"])

    '''
    with accelerator.main_process_first():
        # Log dataset info
        LOGGER.info("Dataset info:")
        info = dataset.info
        size_gb = info.dataset_size / (1024 ** 3)
        num_examples = info.splits['train'].num_examples
        LOGGER.info(f"Dataset Size: {size_gb:.2f} GB")
        LOGGER.info(f"Total Examples: {num_examples:,}") # The :, adds comma separators       
    '''
    # Mix datasets
    # num_fine_web = len(fineweb)
    # num_my_data = len(dataset)
    # total_samples = num_fine_web + num_my_data
    # prob_fine_web = num_fine_web / total_samples
    # prob_my_data = num_my_data / total_samples
    # comment this line to use config probabilities
    # probabilities = [prob_fine_web, prob_my_data]

    probabilities = config["data_mix"]["probabilities"]

    # dataset = dataset.select(range(int(1000 * 0.7)))
    # fineweb = fineweb.select(range(int(1000 * 0.3)))

    combined_dataset = interleave_datasets(
        [fineweb, dataset],
        probabilities=probabilities,
        seed=config["training"]["seed"], 
        stopping_strategy="first_exhausted"
    )
    
    # Training arguments
    training_args = prepare_training_args(config, output_dir)
    resource_logging_steps = config["training"]["resource_logging_steps"]    
    max_length = config["training"]["max_length"]

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        processing_class=tokenizer, # pass tokenizer here
        callbacks=[
            MultiGPUResourcesCallback(resource_logging_steps), 
            ThroughputCallback(resource_logging_steps, max_length, accelerator)
        ]
    )    

    # Train the model
    trainer.train()
    return trainer


def init_wandb(config):
    # Login wandb
    wandb.login()

    # Initialize wandb run
    wandb.init(
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
        }
    )


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
        LOGGER.info("Starting fine-tuning.")
        LOGGER.info(f"Distributed: {accelerator.distributed_type}")
        LOGGER.info(f"Process: {accelerator.process_index}/{accelerator.num_processes}")
        LOGGER.info(f"Config: {config}")
        LOGGER.info(f"Output dir: {output_dir}")
        # Initialize wandb
        if config["training"]["report_to"] == "wandb":
            init_wandb(config)

    # Start training
    trainer = train(config, accelerator, output_dir)
    LOGGER.info(f"[RANK {int(os.environ.get('RANK', 0))}] Finished training")
    
    # Save final model
    final_model_dir = f"{output_dir}/final-model" 
    trainer.save_model(final_model_dir)

    if accelerator.is_main_process:
        LOGGER.info(f"Saved model to {final_model_dir}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune Gemma model.")
    parser.add_argument("--config-path", "-c", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)
    output_dir = args.output_dir
    train_model(config, output_dir)
