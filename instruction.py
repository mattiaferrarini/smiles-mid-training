from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import argparse
from utils.config import load_config, hf_auth 
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig


parser = argparse.ArgumentParser(description="Instruction-Tuning con SFTTrainer.")
parser.add_argument("--config", type=str, required=True, help="Percorso del file di configurazione YAML.")
args = parser.parse_args()

config = load_config(args.config) 
hf_auth() 

model_path = config['model']['name']
data_path = config['data']['data_folder']

dataset = load_dataset(data_path, split="train")
tokenizer = AutoTokenizer.from_pretrained(model_path)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    r=config['peft']['lora_r'],
    lora_alpha=config['peft']['lora_alpha'],
    lora_dropout=config['peft']['lora_dropout'],
    target_modules=config['peft']['lora_target_modules'],
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir=config['training']['output_dir'],
    num_train_epochs=config['training']['epochs'],
    per_device_train_batch_size=config['training']['batch_size'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    learning_rate=float(config['training']['learning_rate']),
    logging_steps=config['training']['logging_steps'],
    save_steps=config['training']['save_steps'],
    bf16=config['training']['bf16'],
    fp16=config['training']['fp16'],
    dataset_text_field="messages", 
    packing=False,
    max_steps=config['training']['max_steps'],
    max_length=2048,
    # just to disable wandb because I am having issues with it
    report_to=config['training'].get('report_to', 'none'), 
    ddp_find_unused_parameters=False
)

trainer = SFTTrainer(
    model=model_path,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config
)


trainer.train()

trainer.save_model(config['training']['output_dir'])
print("Training complete and model saved.")