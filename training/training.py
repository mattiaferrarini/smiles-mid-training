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
    TrainingArguments
)
from datasets import load_dataset
import datasets
from dotenv import load_dotenv
import argparse
from datetime import datetime
from utils.config import load_config
import wandb

# Training hyperparameters
DDP_BACKEND = "nccl"


def train(config):
    # Load tokenizer and model    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype="auto",
        device_map=None # Let accelerator handle device mapping
    )
    
    # TODO: Maybe enable gradient checkpointing
    # model.gradient_checkpointing_enable()
    
    # Load and process dataset
    dataset = load_dataset("arrow", data_dir=config["data"]["data_folder"], data_files="**/*.arrow")
    dataset = dataset["train"]
    
    # TODO: Maybe we need to format examples
    # Tokenize the text field
    dataset = dataset.map(
        lambda x: tokenizer(x[config["data"]["text_field"]]), 
        batched=True,
        num_proc=config["training"]["num_workers"],
	batch_size=5000
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
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
        report_to="wandb",
        # gradient_checkpointing=config["training"]["gradient_checkpointing"],
    )
    
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
    trainer = train(config)
    
    # Save final model
    final_model_dir = f"{output_dir}/final-model"
    if accelerator.is_main_process:
        trainer.save_model(final_model_dir)

 
if __name__ == "__main__":
    main()
