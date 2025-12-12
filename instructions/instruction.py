import os
import sys
import time  # Added for wait loops
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import json
import torch
import random
import argparse

from peft import LoraConfig
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from utils.config import load_config, hf_auth
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from custom_tokenizers.assemble_tokenizer import assemble_tokenizer


def prepare_sciq(output, target_count=15000):
    print("Downloading SciQ dataset...")
    dataset = load_dataset("allenai/sciq", split="train")

    print(f"Processing up to {target_count} examples into strict format...")
    
    # Write to a temporary file first for atomic operation
    tmp_output = output + ".tmp"
    
    with open(tmp_output, 'w') as out:
        count = 0
        for row in dataset:
            if count >= target_count:
                break
            
            question = row['question']
            correct_answer = row['correct_answer']
            distractors = [row['distractor1'], row['distractor2'], row['distractor3']]
            
            options = [correct_answer] + distractors
            random.shuffle(options)
            
            option_text = ""
            correct_label = ""
            labels = ['A', 'B', 'C', 'D']
            
            for i, opt in enumerate(options):
                option_text += f"{labels[i]}. {opt}\n"
                if opt == correct_answer:
                    correct_label = labels[i]
            
            user_prompt = (
                f"{question}\n"
                f"Options:\n{option_text}\n"
                "Answer with the correct letter inside [ANSWER] tags."
            )
            
            assistant_response = f"[ANSWER]{correct_label}[/ANSWER]"
            
            example = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            
            count += 1
            out.write(json.dumps(example) + "\n")

    print(f"Generated {count} examples, saving to {output}...")
    shutil.move(tmp_output, output) # Atomic move

def prepare_metamathqa(output, target_count=15000):
    print("Downloading MetaMathQA dataset...")
    dataset = load_dataset("meta-math/MetaMathQA", split="train")

    print(f"Processing up to {target_count} examples into strict format...")
    
    tmp_output = output + ".tmp"
    
    with open(tmp_output, 'w') as out:
        count = 0
        for row in dataset:
            if count >= target_count:
                break

            question = row.get('query')
            response = row.get('response', '')

            match = re.search(r"The answer is:? (.*?)(?:\.|$)", response)
            if match:
                answer_value = match.group(1).strip()
                user_prompt = (
                    f"{question}\n"
                    "Answer with the numerical value wrapped in [ANSWER] tags."
                )
                strict_response = response.replace(f"The answer is: {answer_value}", f"The answer is: [ANSWER]{answer_value}[/ANSWER]")
                if "[ANSWER]" not in strict_response:
                    strict_response = response + f"\n[ANSWER]{answer_value}[/ANSWER]"
                example = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": strict_response}
                    ]
                }
                
                count += 1
                out.write(json.dumps(example) + "\n")

    print(f"Generated {count} examples, saving to {output}...")
    shutil.move(tmp_output, output) # Atomic move

# def prepare_chemiq_synthetic(output, source_file="chemiq_training_smiles.txt", target_count=20000):
#     print(f"Generating synthetic ChemiQ data from {source_file}...")
    
#     if not os.path.exists(source_file):
#         print(f"Error: {source_file} not found. Skipping ChemiQ generation.")
#         return

#     with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
#         smiles_list = [line.strip() for line in f if line.strip()]
    
#     random.shuffle(smiles_list)

#     # Write to a temporary file first for atomic operation
#     tmp_output = output + ".tmp"

#     # Open with buffering=1 (line buffering) to ensure writes happen
#     with open(tmp_output, 'w', encoding='utf-8') as out:
#         count = 0
#         for smi in smiles_list:
#             if count >= target_count: break
            
#             try:
#                 # Sanitization: Ensure SMILES is a clean string
#                 smi = str(smi)
                
#                 mol = Chem.MolFromSmiles(smi)
#                 if not mol: continue

#                 # CLEANUP: Handle salts
#                 frags = rdmolops.GetMolFrags(mol, asMols=True)
#                 if len(frags) > 1:
#                     mol = max(frags, key=lambda m: m.GetNumAtoms())
                
#                 if mol.GetNumAtoms() < 6: continue

#                 json_record = None

#                 # ---------------------------------------------------------
#                 # TASK 1: SHORTEST PATH
#                 # ---------------------------------------------------------
#                 if random.random() < 0.33:
#                     atoms = list(range(mol.GetNumAtoms()))
#                     a1, a2 = random.sample(atoms, 2)
                    
#                     dist_matrix = rdmolops.GetDistanceMatrix(mol)
#                     dist = int(dist_matrix[a1, a2])
                    
#                     if dist > 1000: continue

#                     mol_copy = Chem.Mol(mol)
#                     mol_copy.GetAtomWithIdx(a1).SetIsotope(99)
#                     mol_copy.GetAtomWithIdx(a2).SetIsotope(98)
                    
#                     tagged_smi = Chem.MolToSmiles(mol_copy, canonical=False)
#                     # Safe regex replacement
#                     prompt_smi = re.sub(r'\[9[89].*?\]', '*', tagged_smi)
                    
#                     if prompt_smi.count('*') == 2:
#                         prompt = (
#                             "Determine the number of bonds along the shortest path connecting the two dummy atoms (denoted by '*'). "
#                             "Count each bond equally, including those directly attached to the dummy atoms.\n\n"
#                             f"{prompt_smi}\n\n"
#                             "Give your answer as an integer. Do not write any comments."
#                         )
#                         response = f"\\boxed{{{dist}}}"
                        
#                         json_record = {
#                             "messages": [
#                                 {"role": "user", "content": prompt},
#                                 {"role": "assistant", "content": response}
#                             ]
#                         }

#                 # ---------------------------------------------------------
#                 # TASK 2: ATOM MAPPING
#                 # ---------------------------------------------------------
#                 elif random.random() < 0.66:
#                     smi1 = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
#                     smi2 = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                    
#                     m1 = Chem.MolFromSmiles(smi1)
#                     m2 = Chem.MolFromSmiles(smi2)
                    
#                     if m1 and m2:
#                         matches = m1.GetSubstructMatch(m2)
#                         if matches and len(matches) == m1.GetNumAtoms():
#                             # Strictly format list of tuples as string
#                             mapping_str = "[" + ", ".join([f"({i}, {match_idx})" for i, match_idx in enumerate(matches)]) + "]"
                            
#                             prompt = (
#                                 "You are given two SMILES strings for the same molecule. Atoms are numbered from left to right, "
#                                 "with the first atom having index 0. Only heavy atoms are numbered and mapped.\n\n"
#                                 f"Molecule 1: {smi1}\n"
#                                 f"Molecule 2: {smi2}\n\n"
#                                 "Determine the mapping of atoms from Molecule 1 to Molecule 2. "
#                                 "Provide your answer as a list of tuples."
#                             )
#                             response = f"\\boxed{{{mapping_str}}}"
                            
#                             json_record = {
#                                 "messages": [
#                                     {"role": "user", "content": prompt},
#                                     {"role": "assistant", "content": response}
#                                 ]
#                             }

#                 # ---------------------------------------------------------
#                 # TASK 3: CARBON COUNTING
#                 # ---------------------------------------------------------
#                 else:
#                     num_c = int(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'))
#                     random_smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                    
#                     prompt = (
#                         "How many carbon atoms are in the molecule:\n\n"
#                         f"{random_smi}\n\n"
#                         "Give your answer as an integer. Do not write any comments."
#                     )
#                     response = f"\\boxed{{{num_c}}}"
                    
#                     json_record = {
#                         "messages": [
#                             {"role": "user", "content": prompt},
#                             {"role": "assistant", "content": response}
#                         ]
#                     }

#                 # --- SAFE WRITE ---
#                 if json_record:
#                     # ensure_ascii=True escapes all non-ASCII chars to \uXXXX, preventing PyArrow parse errors
#                     # separators=(',', ':') removes unnecessary whitespace which can sometimes confuse strict parsers
#                     json_line = json.dumps(json_record, ensure_ascii=True, separators=(',', ':'))
                    
#                     # Double check it is valid JSON
#                     try:
#                         json.loads(json_line)
#                         out.write(json_line + "\n")
#                         count += 1
#                     except Exception:
#                         continue
        
#             except Exception:
#                 continue

#     print(f"Successfully generated {count} valid synthetic ChemiQ examples.")
#     shutil.move(tmp_output, output) # Atomic move

def prepare_datasets(config):
    print("Loading datasets for mixing...")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    chat_dataset = load_dataset("trl-lib/Capybara", split="train")
    print(f"Loaded {len(chat_dataset)} samples from chat dataset")

    # --- SCIQ HANDLING ---
    sciq_path = config.get('data', {}).get('sciq_path', 'sciq.jsonl')
    print(f"Using sciq path: {sciq_path}")
    
    # Only Rank 0 generates data
    if local_rank == 0 and not os.path.exists(sciq_path):
        print(f"{sciq_path} not found, generating from sciq...")
        prepare_sciq(sciq_path)
    
    # Other ranks wait
    if local_rank != 0:
        while not os.path.exists(sciq_path):
            time.sleep(1)
            
    sciq_dataset = load_dataset("json", data_files=sciq_path, split="train")
    print(f"Loaded {len(sciq_dataset)} samples from sciq dataset")

    # --- METAMATHQA HANDLING ---
    metamathqa_path = config.get('data', {}).get('metamathqa_path', 'metamathqa.jsonl')
    print(f"Using methamathqa path: {metamathqa_path}")
    
    if local_rank == 0 and not os.path.exists(metamathqa_path):
        print(f"{metamathqa_path} not found, generating from methamathqa...")
        prepare_metamathqa(metamathqa_path)
        
    if local_rank != 0:
        while not os.path.exists(metamathqa_path):
            time.sleep(1)

    metamathqa_dataset = load_dataset("json", data_files=metamathqa_path, split="train")
    print(f"Loaded {len(metamathqa_dataset)} samples from metamathqa dataset")

    # # --- CHEMIQ HANDLING ---
    # chemiq_path = "chemiq_synthetic.jsonl"
    
    # if local_rank == 0 and not os.path.exists(chemiq_path):
    #     print(f"{chemiq_path} not found, generating synthetic ChemiQ dataset...")
    #     prepare_chemiq_synthetic(chemiq_path, source_file="chemiq_training_smiles.txt", target_count=20000)

    # if local_rank != 0:
    #     while not os.path.exists(chemiq_path):
    #         time.sleep(1)
    
    # chemiq_dataset = load_dataset("json", data_files=chemiq_path, split="train")
    # print(f"Loaded {len(chemiq_dataset)} samples from synthetic ChemiQ dataset")

    print("Creating final dataset...")
    final_dataset = interleave_datasets(
        # [chat_dataset, sciq_dataset, metamathqa_dataset, chemiq_dataset], 
        # probabilities=[0.3, 0.25, 0.25, 0.2], 
        [chat_dataset, sciq_dataset, metamathqa_dataset],
        probabilities=[0.4, 0.3, 0.3],
        seed=42,
        stopping_strategy="first_exhausted"
    )

    # Adding validation set
    # return final_dataset
    dataset_splits = final_dataset.train_test_split(test_size=0.05, seed=42)
    
    print(f"Train size: {len(dataset_splits['train'])}, Validation size: {len(dataset_splits['test'])}")
    return dataset_splits

def setup_tokenizer_and_model(config, args):
    model_path = args.model_path if args.model_path is not None else config['model']['name']
    
    tokenizer = None
    config_found = None
    if os.path.isdir(model_path):
        configs = [
            os.path.join(model_path, "training_config.yaml"),
            os.path.join(os.path.dirname(model_path), "training_config.yaml")
        ]
        
        for cfg in configs:
            if os.path.exists(cfg):
                print(f"Found training config at {cfg}, assembling tokenizer...")
                train_config = load_config(cfg)
                tokenizer = assemble_tokenizer(train_config)
                config_found = cfg
                break
    
    if tokenizer is None:
        print(f"Loading tokenizer from model_path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}
    print(f"Loading model on local rank {local_rank}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float32,
        attn_implementation="eager" 
    )

    special_tokens_dict = {
        "additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    if num_added_toks > 0:
        print(f"Added {num_added_toks} special tokens, resizing model embeddings...")
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying gemma chat template to tokenizer...")
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if (message['role'] == 'user') %}"
        "{{'<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n'}}"
        "{% elif (message['role'] == 'assistant') or (message['role'] == 'model') %}"
        "{{'<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n'}}"
        "{% elif (message['role'] == 'system') %}"
        "{{'<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n'}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{'<start_of_turn>model\n'}}"
        "{% endif %}"
        "{% if not add_generation_prompt %}"
        "{{ eos_token }}"
        "{% endif %}"
    )    
    return tokenizer, model, config_found

# def train(config, model, tokenizer, dataset, base_config):
def train(config, model, tokenizer, dataset_splits, base_config):
    peft_config = LoraConfig(
        r=config['peft']['lora_r'],
        lora_alpha=config['peft']['lora_alpha'],
        lora_dropout=config['peft']['lora_dropout'],
        target_modules=config['peft']['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # training_args = SFTConfig(
    #     output_dir=config['training']['output_dir'],
    #     num_train_epochs=config['training']['epochs'],
    #     per_device_train_batch_size=config['training']['batch_size'],
    #     gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    #     learning_rate=float(config['training']['learning_rate']),
    #     logging_steps=config['training']['logging_steps'],
    #     save_steps=config['training']['save_steps'],
    #     bf16=config['training']['bf16'],
    #     fp16=config['training']['fp16'],
    #     dataset_text_field="messages", 
    #     packing=False,
    #     # max_steps=config['training']['max_steps'],
    #     max_length=2048,
    #     report_to=config['training'].get('report_to', 'none'), 
    #     ddp_find_unused_parameters=False,
    #     gradient_checkpointing=True
    # )
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
        # max_steps=config['training']['max_steps'],
        max_length=2048,
        report_to=config['training'].get('report_to', 'none'), 
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        
        # for validation
        evaluation_strategy="steps",
        eval_steps=config['training']['save_steps'],
        per_device_eval_batch_size=config['training']['batch_size'],
        do_eval=True
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     args=training_args,
    #     peft_config=peft_config,
    #     processing_class=tokenizer
    # )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_splits['train'],
        eval_dataset=dataset_splits['test'],
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    trainer.train()
    print("Training complete, saving model and tokenizer...")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        subdir = f"it-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        final_dir = os.path.join(config['training']['output_dir'], subdir)
        os.makedirs(final_dir, exist_ok=True)

        if base_config and os.path.exists(base_config):
            dest = os.path.join(final_dir, "training_config.yaml")
            shutil.copyfile(base_config, dest)
            print(f"Copied training_config.yaml to {dest}")
        else:
            print("Failed to find training_config.yaml, instruction-tuning might fail...")

        trainer.model.generation_config.eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]
        trainer.model.generation_config.pad_token_id = tokenizer.pad_token_id
        trainer.model.generation_config.save_pretrained(final_dir)

        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

    return trainer

def main():
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    parser = argparse.ArgumentParser(description="Instruction-Tuning con SFTTrainer.")
    parser.add_argument("--config", type=str, required=True, help="Percorso del file di configurazione YAML.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional: override model name/path from config.")
    args = parser.parse_args()
    
    config = load_config(args.config) 
    hf_auth() 

    # dataset = prepare_datasets(config)
    dataset_splits = prepare_datasets(config)

    tokenizer, model, base_config = setup_tokenizer_and_model(config, args)
    # trainer = train(config, model, tokenizer, dataset, base_config)
    trainer = train(config, model, tokenizer, dataset_splits, base_config)

    print("\n=== SANITY CHECK: TEST GENERATION ===")
    test_messages = [{"role": "user", "content": "Explain what a molecule is in one sentence."}]

    prompt_str = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    print(f"Test Input: {prompt_str}")

    inputs = tokenizer(prompt_str, return_tensors="pt").to(trainer.model.device)
    model_to_gen = trainer.accelerator.unwrap_model(trainer.model)
    model_to_gen.eval()

    with torch.no_grad():
        outputs = model_to_gen.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            eos_token_id=trainer.model.generation_config.eos_token_id
        )

    print(f"Model Output:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()

