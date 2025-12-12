import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import torch
from rdkit import RDLogger
from datetime import timedelta
from peft import PeftModel, PeftConfig
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import load_config
from custom_tokenizers.hybrid_tokenizer import HybridTokenizer
from custom_tokenizers.assemble_tokenizer import assemble_tokenizer

from chembench.utils import enable_logging
from chembench.prompter import PrompterBuilder
from chembench.types import Generation, Generations
from chembench.evaluate import ChemBenchmark, save_topic_reports

from ChemIQ.utils.parser import AnswerParser
from ChemIQ.utils.answer_verifier import AnswerVerifier

RDLogger.DisableLog('rdApp.*')
AutoTokenizer.register("HybridTokenizer", HybridTokenizer)

CHEMBENCH_TOPICS = [
    'analytical_chemistry', 
    'chemical_preference', 
    'general_chemistry', 
    'inorganic_chemistry', 
    'materials_science', 
    'organic_chemistry', 
    'physical_chemistry', 
    'technical_chemistry', 
    'toxicity_and_safety'
]

class GemmaWrapper:
    """
    Wrapper for loading gemma-3-1b models and their fine-tuned variants for chemical benchmarks
    """
    def __init__(self, model_path, device=None):
        """
        Initializes the GemmaWrapper

        Args:
            model_path (str): Path to the pretrained or fine-tuned model (weights or adapter config)
            device (torch.device, optional): The device to load the model on
        """
        self.device = device
        device_map = {"": device} if device is not None else "auto"

        print(f"Loading tokenizer from: {model_path}")

        config_path = None        
        search_paths = [
            os.path.join(model_path, "training_config.yaml"),
            os.path.join(os.path.dirname(model_path), "training_config.yaml")
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        self.tokenizer = None
        if config_path:
            print(f"Found training config at {config_path}, assembling tokenizer...")
            try:
                train_config = load_config(config_path)
                self.tokenizer = assemble_tokenizer(train_config)
            except Exception as e:
                print(f"Failed to assemble tokenizer from config: {e}")

        if self.tokenizer is None:
            print(f"Falling back to AutoTokenizer.from_pretrained for {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        try:
            config = PeftConfig.from_pretrained(model_path)
            base_model_path = config.base_model_name_or_path
            print(f"Found base model path from PEFT config: {base_model_path}")
            is_peft = True
        except Exception:
            base_model_path = model_path
            print(f"No PEFT config found, using full model: {base_model_path}")
            is_peft = False

        print("Loading base model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            device_map=device_map, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )

        current_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > current_vocab_size:
            print(f"Resizing model embeddings from {current_vocab_size} to {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))

        if is_peft:
            print(f"Loading adapter weights from: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)

        if not self.tokenizer.chat_template:
            print("Chat template missing, applying gemma default template...")
            self.tokenizer.chat_template = (
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

    def format_prompt(self, messages):
        """
        Applies the chat template to a list of messages

        Args:
            messages (list): List of message dictionaries (role, content)
        
        Returns:
            str: Formatted prompt string
        """
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, **kwargs):
        """
        Generates text based on the provided prompts using the loaded Gemma model

        Args:
            prompts (str or list): Input prompt or list of prompts
            **kwargs: Additional arguments

        Returns:
            Generations: A ChemBench Generations object containing the results
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        formatted_prompts = []
        for prompt in prompts:
            if isinstance(prompt, str):
                formatted_prompts.append(self.format_prompt([{"role": "user", "content": prompt}]))
            elif isinstance(prompt, list):
                formatted_prompts.append(self.format_prompt(prompt))
        
        inputs = self.tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        cleaned_outputs = [text.strip() for text in decoded_outputs]
        generations = [[Generation(text=text)] for text in cleaned_outputs]
        return Generations(generations=generations)  

def run_chembench(model_path, output_path):
    """
    Runs the ChemBench benchmark suite on the specified Gemma model

    Args:
        model_path (str): Path to the Gemma model/adapter
        output_path (str): Directory to save results
    """
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[timeout])
    
    accelerator.wait_for_everyone()
    print(f"[Init] Rank: {accelerator.process_index}/{accelerator.num_processes} | Device: {accelerator.device}", flush=True)

    enable_logging()

    model = GemmaWrapper(model_path, device=accelerator.device)  
    prompter = PrompterBuilder.from_model_object(model=model, prompt_type="instruction")

    my_topics = [
        topic for i, topic in enumerate(CHEMBENCH_TOPICS) 
        if i % accelerator.num_processes == accelerator.process_index
    ]
    print(f"[Rank {accelerator.process_index}] Processing {len(my_topics)} topics: {my_topics}")

    benchmark = ChemBenchmark.from_huggingface()
    if len(my_topics) > 0:
        results = benchmark.bench(prompter, batch_size=8, topics=my_topics)

        os.makedirs(output_path, exist_ok=True)
        save_topic_reports(benchmark, results, output_path)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("All ranks are done")

    # benchmark.submit(results)

def run_chemiq(model_path, chemiq_path, output_path):
    """
    Runs the ChemIQ benchmark on the specified Gemma model

    Args:
        model_path (str): Path to the Gemma model/adapter
        chemiq_path (str): Root directory of the ChemIQ repository
        output_path (str): Directory to save results
    """
    questions = []
    jsonl_path = os.path.join(chemiq_path, "questions", "chemiq.jsonl")
    with open(jsonl_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))

    model = GemmaWrapper(model_path, device="cuda:0")
    parser = AnswerParser(jsonl_path) 
    verifier = AnswerVerifier(jsonl_path)

    os.makedirs(output_path, exist_ok=True)
    results_file = os.path.join(output_path, "chemiq_results.jsonl")

    results = []
    for i, question in enumerate(questions):
        id = question.get("uuid")
        prompt = question.get("prompt")
        messages = [{"role": "user", "content": prompt}]

        print(f"Processing ChemiQ question ({i}/{len(questions)}) {id}...")
        try:
            generations = model.generate([messages])
            
            raw_answer = None
            if generations and hasattr(generations, 'generations') and len(generations.generations) > 0:
                first_gen = generations.generations[0]
                if first_gen and len(first_gen) > 0:
                    raw_answer = first_gen[0].text
            
            if not raw_answer:
                print(f"Warning: Model generated empty output for {id}")
                raw_answer = ""

            if raw_answer:
                parsed_answer = parser.parse(id, raw_answer)
            else:
                parsed_answer = None

            record = {
                "uuid": id,
                "question_category": question.get("question_category"),
                "sub_category": question.get("sub_category"),
                "prompt": prompt,
                "expected_answer": question.get("answer"),
                "verification_method": question.get("verification_method"),
                "model_answer": raw_answer,
            }

            if parsed_answer:
                try:
                    check = verifier.check_answer(id, parsed_answer)
                    record.update(check)
                except Exception as e:
                    print(f"Failed to verify answer for {id}: {e}")
                    record.update({"is_correct": False, "opsin_smiles": None})
            else:
                record.update({"is_correct": False if raw_answer else None, "opsin_smiles": None})

            results.append(record)
            with open(results_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
        
        except Exception as e:
            print(f"CRITICAL ERROR on question {id}: {e}")
            continue

    summary = {
        "num_questions": len(questions),
        "num_processed": len(results),
        "num_attempted": sum(1 for r in results if r.get("is_correct") is not None),
        "num_correct": sum(1 for r in results if r.get("is_correct") is True),
        "accuracy": 0.0,
    }

    if summary["num_attempted"] > 0:
        summary["accuracy"] = summary["num_correct"] / summary["num_attempted"]

    summary_path = os.path.join(output_path, "chemiq_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
