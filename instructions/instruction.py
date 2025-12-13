import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import json
import time
import torch
import random
import shutil

from peft import LoraConfig
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from utils.config import load_config, hf_auth
from utils.logging import get_logger
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from custom_tokenizers.assemble_tokenizer import assemble_tokenizer

LOGGER = get_logger(__name__)


def prepare_sciq(output, target_count=15000):
    LOGGER.info("Downloading SciQ dataset...")
    dataset = load_dataset("allenai/sciq", split="train")

    LOGGER.info(f"Processing up to {target_count} examples into strict format...")

    # Write to a temporary file first for atomic operation
    tmp_output = output + ".tmp"

    with open(tmp_output, "w") as out:
        count = 0
        for row in dataset:
            if count >= target_count:
                break

            question = row["question"]
            correct_answer = row["correct_answer"]
            distractors = [row["distractor1"], row["distractor2"], row["distractor3"]]

            options = [correct_answer] + distractors
            random.shuffle(options)

            option_text = ""
            correct_label = ""
            labels = ["A", "B", "C", "D"]

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
                    {"role": "assistant", "content": assistant_response},
                ]
            }

            count += 1
            out.write(json.dumps(example) + "\n")

    LOGGER.info(f"Generated {count} examples, saving to {output}...")
    shutil.move(tmp_output, output)  # Atomic move


def prepare_metamathqa(output, target_count=15000):
    LOGGER.info("Downloading MetaMathQA dataset...")
    dataset = load_dataset("meta-math/MetaMathQA", split="train")

    LOGGER.info(f"Processing up to {target_count} examples into strict format...")

    tmp_output = output + ".tmp"

    with open(tmp_output, "w") as out:
        count = 0
        for row in dataset:
            if count >= target_count:
                break

            question = row.get("query")
            response = row.get("response", "")

            match = re.search(r"The answer is:? (.*?)(?:\.|$)", response)
            if match:
                answer_value = match.group(1).strip()
                user_prompt = (
                    f"{question}\n"
                    "Answer with the numerical value wrapped in [ANSWER] tags."
                )
                strict_response = response.replace(
                    f"The answer is: {answer_value}",
                    f"The answer is: [ANSWER]{answer_value}[/ANSWER]",
                )
                if "[ANSWER]" not in strict_response:
                    strict_response = response + f"\n[ANSWER]{answer_value}[/ANSWER]"
                example = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": strict_response},
                    ]
                }

                count += 1
                out.write(json.dumps(example) + "\n")

    LOGGER.info(f"Generated {count} examples, saving to {output}...")
    shutil.move(tmp_output, output)  # Atomic move


def prepare_datasets(config):
    LOGGER.info("Loading datasets for mixing...")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    chat_dataset = load_dataset("trl-lib/Capybara", split="train")
    LOGGER.info(f"Loaded {len(chat_dataset)} samples from chat dataset")

    sciq_path = os.path.expandvars(config.get("data", {}).get("sciq_path", "sciq.jsonl"))
    LOGGER.info(f"Using sciq path: {sciq_path}")

    # Only Rank 0 generates data
    if local_rank == 0 and not os.path.exists(sciq_path):
        LOGGER.info(f"{sciq_path} not found, generating from sciq...")
        prepare_sciq(sciq_path)

    # Other ranks wait
    if local_rank != 0:
        while not os.path.exists(sciq_path):
            time.sleep(1)

    sciq_dataset = load_dataset("json", data_files=sciq_path, split="train")
    LOGGER.info(f"Loaded {len(sciq_dataset)} samples from sciq dataset")

    metamathqa_path = os.path.expandvars(config.get("data", {}).get("metamathqa_path", "metamathqa.jsonl"))
    LOGGER.info(f"Using methamathqa path: {metamathqa_path}")

    if local_rank == 0 and not os.path.exists(metamathqa_path):
        LOGGER.info(f"{metamathqa_path} not found, generating from methamathqa...")
        prepare_metamathqa(metamathqa_path)

    if local_rank != 0:
        while not os.path.exists(metamathqa_path):
            time.sleep(1)

    metamathqa_dataset = load_dataset("json", data_files=metamathqa_path, split="train")
    LOGGER.info(f"Loaded {len(metamathqa_dataset)} samples from metamathqa dataset")

    LOGGER.info("Creating final dataset...")
    final_dataset = interleave_datasets(
        [chat_dataset, sciq_dataset, metamathqa_dataset],
        probabilities=[0.4, 0.3, 0.3],
        seed=42,
        stopping_strategy="first_exhausted",
    )

    dataset_splits = final_dataset.train_test_split(test_size=0.05, seed=42)

    LOGGER.info(
        f"Train size: {len(dataset_splits['train'])}, Validation size: {len(dataset_splits['test'])}"
    )
    return dataset_splits


def setup_tokenizer_and_model(config, model_path_override=None):
    model_path = (
        model_path_override
        if model_path_override is not None
        else config["model"]["name"]
    )

    tokenizer = None
    config_found = None
    if os.path.isdir(model_path):
        configs = [
            os.path.join(model_path, "training_config.yaml"),
            os.path.join(os.path.dirname(model_path), "training_config.yaml"),
        ]

        for cfg in configs:
            if os.path.exists(cfg):
                LOGGER.info(f"Found training config at {cfg}, assembling tokenizer...")
                train_config = load_config(cfg)
                tokenizer = assemble_tokenizer(train_config)
                config_found = cfg
                break

    if tokenizer is None:
        LOGGER.info(f"Loading tokenizer from model_path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}
    LOGGER.info(f"Loading model on local rank {local_rank}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float32,
        attn_implementation="eager",
    )

    special_tokens_dict = {
        "additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    if num_added_toks > 0:
        LOGGER.info(
            f"Added {num_added_toks} special tokens, resizing model embeddings..."
        )
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Applying gemma chat template to tokenizer...")
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


def train(config, model, tokenizer, dataset_splits, base_config):
    peft_config = LoraConfig(
        r=config["peft"]["lora_r"],
        lora_alpha=config["peft"]["lora_alpha"],
        lora_dropout=config["peft"]["lora_dropout"],
        target_modules=config["peft"]["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=os.path.expandvars(config["training"]["output_dir"]),
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
        dataset_text_field="messages",
        packing=False,
        # max_steps=config['training']['max_steps'],
        max_length=2048,
        report_to=config["training"].get("report_to", "none"),
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        # for validation
        evaluation_strategy="steps",
        eval_steps=config["training"]["save_steps"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        do_eval=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    LOGGER.info("Training complete, saving model and tokenizer...")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        subdir = f"it-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        final_dir = os.path.join(os.path.expandvars(config["training"]["output_dir"]), subdir)
        os.makedirs(final_dir, exist_ok=True)

        if base_config and os.path.exists(base_config):
            dest = os.path.join(final_dir, "training_config.yaml")
            shutil.copyfile(base_config, dest)
            LOGGER.info(f"Copied training_config.yaml to {dest}")
        else:
            LOGGER.warning(
                "Failed to find training_config.yaml, instruction-tuning might fail..."
            )

        trainer.model.generation_config.eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]
        trainer.model.generation_config.pad_token_id = tokenizer.pad_token_id
        trainer.model.generation_config.save_pretrained(final_dir)

        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

    return trainer


def run_instruction_tuning(config_path, model_path=None):
    """
    Executes instruction tuning based on the provided configuration

    Args:
        config_path (str): Path to the YAML configuration file
        model_path (str, optional): Override model name/path from config
    """
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    config = load_config(config_path)
    hf_auth()

    dataset_splits = prepare_datasets(config)

    tokenizer, model, base_config = setup_tokenizer_and_model(config, model_path)
    trainer = train(config, model, tokenizer, dataset_splits, base_config)

    LOGGER.info("\nSanity Check")
    test_messages = [
        {"role": "user", "content": "Explain what a molecule is in one sentence."}
    ]

    prompt_str = tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    LOGGER.info(f"Test Input: {prompt_str}")

    inputs = tokenizer(prompt_str, return_tensors="pt").to(trainer.model.device)
    model_to_gen = trainer.accelerator.unwrap_model(trainer.model)
    model_to_gen.eval()

    with torch.no_grad():
        outputs = model_to_gen.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            eos_token_id=trainer.model.generation_config.eos_token_id,
        )

    LOGGER.info(
        f"Model Output:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}"
    )
