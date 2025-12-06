from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import argparse
from utils.config import load_config, hf_auth 
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig
import os
import torch
from custom_tokenizers.hybrid_tokenizer import HybridTokenizer

parser = argparse.ArgumentParser(description="Instruction-Tuning con SFTTrainer.")
parser.add_argument("--config", type=str, required=True, help="Percorso del file di configurazione YAML.")
args = parser.parse_args()

config = load_config(args.config) 
hf_auth() 

model_path = config['model']['name']
data_path = config['data']['data_folder']
# TODO: write specific tokenizer path

dataset = load_dataset(data_path, split="train")

# --- Tokenizer Setup ---
print("Loading Base Tokenizer...")
base_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Check for custom BPE tokenizer
chem_tokenizer_path = "custom_tokenizers/smiles_bpe/tokenizer.json"
if os.path.exists(chem_tokenizer_path):
    print(f"Loading Chemical BPE Tokenizer from {chem_tokenizer_path}...")
    chem_tokenizer = PreTrainedTokenizerFast(tokenizer_file=chem_tokenizer_path)
    
    print("Initializing Hybrid Tokenizer...")
    tokenizer = HybridTokenizer(
        base_tokenizer=base_tokenizer,
        chem_tokenizer=chem_tokenizer,
        chem_start="[START_SMILES]",
        chem_end="[END_SMILES]"
    )
else:
    print(f"Warning: Chemical tokenizer not found at {chem_tokenizer_path}. Using base tokenizer only.")
    tokenizer = base_tokenizer

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}

print(f"Loading model on Local Rank: {local_rank}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map, # Crucial for DDP
    torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float32,
    attn_implementation="eager" # TODO: set "flash_attention_2" if available 
)
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
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer
)


trainer.train()

if local_rank == 0:
    trainer.save_model(config['training']['output_dir'])
    print("Training complete and model saved.")

    # --- SANITY CHECK: Test Generation ---
    print("\n=== SANITY CHECK: TEST GENERATION ===")
    # Use a simple prompt to see if the model follows instructions
    test_messages = [{"role": "user", "content": "Explain what a molecule is in one sentence."}]
    
    # Apply the chat template (crucial to verify formatting)
    try:
        prompt_str = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"Warning: Could not apply chat template ({e}). Using raw prompt.")
        prompt_str = "User: Explain what a molecule is in one sentence.\nAssistant:"

    print(f"Test Input: {prompt_str}")
    
    inputs = tokenizer(prompt_str, return_tensors="pt").to(trainer.model.device)
    
    # Unwrap model to ensure .generate() works correctly in DDP
    model_to_gen = trainer.accelerator.unwrap_model(trainer.model)

    model_to_gen.eval() 
    
    # Generate
    with torch.no_grad():
        outputs = model_to_gen.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7)
    
    print(f"Model Output:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
