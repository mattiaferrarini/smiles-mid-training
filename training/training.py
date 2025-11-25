import os
import sys
from pathlib import Path

# Add parent directory to Python path to enable imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset
import datasets
from dotenv import load_dotenv
import argparse
from datetime import datetime
from utils.config import load_config
import wandb
import torch
import time


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


class ThroughputLoggerCallback(TrainerCallback):
    def __init__(self, log_steps):
        super().__init__()
        self.log_steps = log_steps
        self.last_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_steps == 0 and state.global_step > 0:
            current_time = time.time()
            time_delta = current_time - self.last_time
            
            # Calculate global effective batch size
            total_batch_size = (
                args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
            )
            
            samples_processed = total_batch_size * self.log_steps
            samples_per_sec = samples_processed / time_delta
            
            self.last_time = current_time

            if args.process_index == 0:
                # Log to wandb and terminal
                if wandb.run is not None:
                    wandb.log(
                        {"throughput/samples_per_sec": samples_per_sec}, 
                        step=state.global_step
                    )
                print(f"[Step {state.global_step}] Throughput: {samples_per_sec:.2f} samples/sec")


def prepare_training_args(config, output_dir):
    strategy = config["distributed"]["strategy"]

    # Common arguments
    args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["training"]["per_device_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "num_train_epochs": config["training"]["epochs"],
        "warmup_steps": config["training"]["warmup_steps"],
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
        "dataloader_persistent_workers": True
    }

    if strategy == "fsdp":
        # FSDP-specific arguments
        fsdp_conf = config["distributed"]["fsdp"]
        args_dict["fsdp"] = fsdp_conf["policy"]
        fsdp_inner_config = fsdp_conf["config"].copy()

        if "fsdp_transformer_layer_cls_to_wrap" in fsdp_inner_config:
            args_dict["fsdp_transformer_layer_cls_to_wrap"] = fsdp_inner_config.pop("fsdp_transformer_layer_cls_to_wrap")
            print(f"FSDP Wrapping Layer: {args_dict['fsdp_transformer_layer_cls_to_wrap']}")
        
        if args_dict["gradient_checkpointing"]:
            fsdp_inner_config["activation_checkpointing"] = True
        args_dict["gradient_checkpointing"] = False       
 
        args_dict["fsdp_config"] = fsdp_inner_config
        
        print(f"Training Strategy: FSDP ({args_dict['fsdp']})")
    elif strategy == "ddp":
        # DDP-specific arguments
        ddp_conf = config["distributed"]["ddp"]
        args_dict["ddp_backend"] = ddp_conf["backend"]
        args_dict["ddp_find_unused_parameters"] = ddp_conf["find_unused_parameters"]
        print(f"Training Strategy: DDP (Backend: {args_dict['ddp_backend']})")
    else:
        raise ValueError(f"Unsupported distributed strategy: {strategy}")
    
    training_args = TrainingArguments(**args_dict)
    return training_args

def train(config, accelerator, output_dir):
    # Load tokenizer   
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load mode
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=torch.bfloat16,
        device_map=None # Let accelerator handle device mapping
    )

    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    # Load and process dataset
    dataset = load_dataset(
        "arrow", 
        data_dir=config["data"]["data_folder"], 
        data_files="**/*.arrow", 
        # streaming=True
    )
    dataset = dataset["train"]
    # TODO: remove this line for full training
    dataset = dataset.select(range(10000))

    # Tokenize the text field
    # Silence progress bars on non-main processes
    if not accelerator.is_main_process:
        datasets.utils.logging.disable_progress_bar()

    def tokenize_and_split(examples):
        outputs = tokenizer(
            examples[config["data"]["text_field"]],
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
        dataset = dataset.map(
            tokenize_and_split,
            batched=True,
            num_proc=config["training"]["num_workers"],
            batch_size=10000,
            remove_columns=dataset.column_names, 
            load_from_cache_file=True
        )

    # Re-enable logging
    if not accelerator.is_main_process:
        datasets.utils.logging.enable_progress_bar()   
 
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=64
    )
    
    # Training arguments
    training_args = prepare_training_args(config, output_dir)
    resource_logging_steps = config["training"]["resource_logging_steps"]    

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            MultiGPUResourcesCallback(resource_logging_steps), 
            ThroughputLoggerCallback(resource_logging_steps)
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


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune Gemma model.")
    parser.add_argument("--config-path", "-c", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)
    output_dir = args.output_dir

    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # Initialize distributed training
    accelerator = Accelerator()

    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("Starting fine-tuning.")
        print(f"Distributed: {accelerator.distributed_type}")
        print(f"Process: {accelerator.process_index}/{accelerator.num_processes}")
        print(f"Config: {config}")
        print(f"Output dir: {output_dir}")

        # Initialize wandb
        if config["training"]["report_to"] == "wandb":
            init_wandb(config)

    # Start training
    trainer = train(config, accelerator, output_dir)
    print(f"[RANK {int(os.environ.get("RANK", 0))}] Finished training")
    
    # Save final model
    final_model_dir = f"{output_dir}/final-model" 
    trainer.save_model(final_model_dir)

    if accelerator.is_main_process:
        print(f"Saved model to {final_model_dir}")
 
if __name__ == "__main__":
    main()
