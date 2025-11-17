from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from dotenv import load_dotenv
import argparse
from datetime import datetime
from pathlib import Path

# Training hyperparameters
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 2
WARMUP_STEPS = 100
LEARNING_RATE = 2e-5
SAVE_STEPS = 1000
LOGGING_STEPS = 50
BF16 = True
FP16 = False
DATA_LOADER_NUM_WORKERS = 4
DDP_BACKEND = "nccl"


def train(model_name, data_folder, output_dir, text_field):
    # Load tokenizer and model    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None # Let accelerator handle device mapping
    )
    
    # TODO: Maybe enable gradient checkpointing
    # model.gradient_checkpointing_enable()
    
    # Load and process dataset
    dataset = load_dataset("arrow", data_dir=data_folder)
    dataset = dataset["train"]
    
    # TODO: Maybe we need to format examples
    # Tokenize the text field
    dataset = dataset.map(lambda x: tokenizer(x[text_field]), batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        bf16=BF16,
        fp16=FP16,
        remove_unused_columns=False,
        dataloader_num_workers=DATA_LOADER_NUM_WORKERS,
        ddp_backend=DDP_BACKEND,
        report_to="none", # TODO: report to wandb
        # gradient_checkpointing=True,
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


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune Gemma model.")
    parser.add_argument("-m", "--model-name", type=str, required=True)
    parser.add_argument("-d", "--data-folder", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("-t", "--text-field", type=str, default="text")
    args = parser.parse_args()

    model_name = args.model_name
    data_folder = args.data_folder
    output_dir = args.output_dir
    text_field = args.text_field

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(output_dir) / timestamp)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize distributed training
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("Starting fine-tuning.")
        print(f"Model name: {model_name}")
        print(f"Data folder: {data_folder}")
        print(f"Output directory: {output_dir}")
        print(f"Training on text field: {text_field}")
        print(f"Distributed: {accelerator.distributed_type}")
        print(f"Process: {accelerator.process_index}/{accelerator.num_processes}")

    # Start training
    trainer = train(model_name, data_folder, output_dir, text_field)
    
    # Save final model
    final_model_dir = f"{output_dir}/final-model"
    if accelerator.is_main_process:
        trainer.save_model(final_model_dir)

if __name__ == "__main__":
    main()