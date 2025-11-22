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

# Training hyperparameters
DDP_BACKEND = "nccl"


def train(config, accelerator, output_dir):
    # Load tokenizer and model    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=torch.bfloat16,
        device_map=None # Let accelerator handle device mapping
    )
    
    class MultiGPUResourcesCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Print memory usage for EVERY rank to ensure load balancing
            if torch.cuda.is_available():
                rank = int(os.environ.get("RANK", 0))
                # Only print for all ranks occasionally (e.g. every 10 steps) to reduce log spam
                # Or every step if you are debugging short runs
                if state.global_step % 1 == 0: 
                    current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(f"[Step {state.global_step}] Rank {rank}: {current_mem:.2f} GB (Max: {max_mem:.2f} GB)")

    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    # Load and process dataset
    dataset = load_dataset("arrow", data_dir=config["data"]["data_folder"], data_files="**/*.arrow")
    dataset = dataset["train"]
    dataset = dataset.select(range(200))
    
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
            stride=config["training"]["stride_size"] # Overlap between chunks
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
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"]["per_device_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["epochs"],
        warmup_steps=config["training"]["warmup_steps"],
        learning_rate=config["training"]["learning_rate"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
        remove_unused_columns=False,
        dataloader_num_workers=config["training"]["num_workers"],
        ddp_backend=DDP_BACKEND,
	    ddp_find_unused_parameters=True,
        report_to="wandb",
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[MultiGPUResourcesCallback()]
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
    
    if accelerator.is_main_process:
        print("Starting fine-tuning.")
        print(f"Distributed: {accelerator.distributed_type}")
        print(f"Process: {accelerator.process_index}/{accelerator.num_processes}")
        print(f"Config: {config}")
        print(f"Output dir: {output_dir}")
	
	# Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(output_dir) / timestamp)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        init_wandb(config)

    # Start training
    trainer = train(config, accelerator, output_dir)
    
    # Save final model
    final_model_dir = f"{output_dir}/final-model"
    if accelerator.is_main_process:
        trainer.save_model(final_model_dir)

 
if __name__ == "__main__":
    main()
