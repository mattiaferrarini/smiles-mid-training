import os
import sys
from pathlib import Path

# Add parent directory to Python path to enable imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk
import numpy as np
from tqdm import tqdm
import argparse
import json
from utils.config import load_config
from custom_tokenizers.assemble_tokenizer import assemble_tokenizer
from utils.create_smiles import annotate_smiles

from custom_tokenizers.hybrid_tokenizer import HybridTokenizer
AutoTokenizer.register("HybridTokenizer", HybridTokenizer)

DATASET = "/capstor/store/cscs/swissai/a131/ML4Science/datasets/chembench_mcq/"
HF_DATASET = "jablonkagroup/ChemBench"

configs = ['analytical_chemistry', 'chemical_preference', 'general_chemistry', 
           'inorganic_chemistry', 'materials_science', 'organic_chemistry', 
           'physical_chemistry', 'technical_chemistry', 'toxicity_and_safety']


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def find_checkpoint_folders(path):
    if not os.path.isdir(path):
        return [path]
    
    # Check if this path itself is a valid model (has config.json)
    if os.path.exists(os.path.join(path, "config.json")):
        return [path]
    
    # Look for checkpoint folders
    checkpoints = []
    for item in sorted(os.listdir(path)):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            # Check if it's a valid checkpoint (has config.json)
            if os.path.exists(os.path.join(item_path, "config.json")):
                checkpoints.append(item_path)
    
    # If no checkpoints found, return the original path
    return checkpoints if checkpoints else [path]


def get_likelihood(model, tokenizer, question, answer, debug=False, use_smiles_annotation=False):
    if use_smiles_annotation:
        # Annotate question and answer with [START_SMILES] tags
        question, _ = annotate_smiles(question)
        answer, _ = annotate_smiles(answer)

    prompt_text = f"Question: {question}\nAnswer:"
    # Add space before answer to ensure it tokenizes separately
    full_text = prompt_text + " " + answer
    
    # Tokenize prompt + answer
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # Tokenize prompt only to find its length
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    
    # Debug: Print tokenization info
    if debug:
        print(f"  Debug - Full text length: {input_ids.shape[1]}, Prompt length: {prompt_len}, Answer: '{answer}'")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Slice to only include answer part
    answer_start = prompt_len - 1 
    if answer_start >= shift_labels.shape[1]:
        return -float('inf') 
    
    answer_logits = shift_logits[:, answer_start:, :]
    answer_labels = shift_labels[:, answer_start:]
    
    # Check if answer portion is empty
    if answer_labels.numel() == 0:
        return -float('inf')
    
    # Compute negative log likelihood
    # Mean to avoid penalizing longer answers
    loss_fct = CrossEntropyLoss(reduction="mean")
    neg_log_likelihood = loss_fct(
        answer_logits.view(-1, answer_logits.size(-1)), 
        answer_labels.view(-1)
    )
    
    return -neg_log_likelihood.item()


def process_example(example):
    question = example.get("input")
    raw_scores = example.get("target_scores")
    
    if not question or not raw_scores:
        return None, None, None

    if isinstance(raw_scores, str):
        try:
            target_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            return None, None, None
    elif isinstance(raw_scores, dict):
        target_scores = raw_scores
    else:
        return None, None, None
    
    candidates = list(target_scores.keys())
    if not candidates:
        return None, None, None
    
    return question, target_scores, candidates


def eval_config(model, tokenizer, config, debug=False, use_smiles_annotation=False):
    if os.path.exists(DATASET):
        ds = load_from_disk(DATASET)[config]
    else:
        ds = load_dataset(HF_DATASET, config)["train"]
        ds = ds.filter(lambda x: x["metrics"] == ["multiple_choice_grade"])
    
    print(f"Evaluating {config}: {len(ds)} questions.")
    
    correct_count = 0
    total_count = 0
    
    for idx in tqdm(range(len(ds)), desc=f"{config}"):
        row = ds[idx]
        examples_list = row.get("examples")
        
        if not examples_list:
            continue
            
        for example in examples_list:
            question, target_scores, candidates = process_example(example)
            if question is None or target_scores is None or candidates is None:
                continue

            candidate_scores = []

            if debug:
                print(f"Question: {question}")
            # Calculate likelihoods
            for candidate in candidates:
                score = get_likelihood(model, tokenizer, question, candidate, debug=debug, use_smiles_annotation=use_smiles_annotation)
                if debug:
                    print(f"Candidate: {candidate}, Score: {score}")
                candidate_scores.append(score)

            # Select best candidate
            if debug:
                print("Candidate scores:", candidate_scores)
                print("Target scores:", target_scores)

            best_idx = np.argmax(candidate_scores)
            predicted_answer = candidates[best_idx]
            if debug:
                print(f"Predicted answer: {predicted_answer}")

            # Check correctness
            if target_scores[predicted_answer] == 1:
                correct_count += 1
            total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy for {config}: {accuracy:.2%} ({correct_count}/{total_count})")
    return {"config": config, "accuracy": accuracy, "correct": correct_count, "total": total_count}


def eval_path(model_path, tokenizer, local_rank, debug=False, use_smiles_annotation=False):
    # Load model on the correct device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.eval()

    # Evaluate each config
    results = []
    for config in configs:
        result = eval_config(model, tokenizer, config, debug=debug, use_smiles_annotation=use_smiles_annotation)
        results.append(result)
    
    # Aggregate overall accuracy
    correct, total = 0, 0
    for res in results:
        correct += res['correct']
        total += res['total']
    overall_accuracy = correct / total if total > 0 else 0
    
    return {"path": model_path, "results": results, "overall_accuracy": overall_accuracy, "correct": correct, "total": total}


def log_path_results(eval_results):
    path = eval_results['path']
    results = eval_results['results']
    overall_accuracy = eval_results['overall_accuracy']
    correct = eval_results['correct']
    total = eval_results['total']

    print(f"Results for model at {path}:")
    print("=" * 20)
    for res in results:
        print(f"Config: {res['config']}, Accuracy: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
    print("=" * 20)
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
    print("=" * 20)


def log_all_results(all_results):
    print("\nSummary of all evaluated checkpoints:")

    # Headers for the table
    headers = ["Model"] + configs + ["Overall"]
    
    # Prepare rows
    rows = []
    for result in all_results:
        path = result['path']
        short_path = os.path.basename(os.path.normpath(path))
        row_data = [short_path]
        
        config_map = {item['config']: item['accuracy'] for item in result['results']}
        
        for config in configs:
            acc = config_map.get(config, 0.0)
            row_data.append(f"{acc:.2%}")
        
        row_data.append(f"{result['overall_accuracy']:.2%}")
        rows.append(row_data)

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    col_widths = [w + 2 for w in col_widths]

    # Formatter string
    header_fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    row_fmt = header_fmt

    # Print table
    print("=" * sum(col_widths))
    print(header_fmt.format(*headers))
    print("-" * sum(col_widths))
    for row in rows:
        print(row_fmt.format(*row))
        print("-" * sum(col_widths))
    print("=" * sum(col_widths))

    # Find and log best model
    best_result = max(all_results, key=lambda x: x['overall_accuracy'])
    print("\n" + "="*50)
    print(f"BEST MODEL (Overall Accuracy): {os.path.basename(os.path.normpath(best_result['path']))}")
    print("="*50)
    log_path_results(best_result)
    

def get_tokenizer_for_eval(model_path):
    if os.path.exists(model_path) and os.path.isdir(model_path):
        # Local directory
        if os.path.exists(os.path.join(model_path, "config.json")):
            # Single checkpoint folder: find config in parent
            training_config_path = os.path.join(os.path.dirname(model_path), "training_config.yaml")    
        else:
            # Folder with multiple checkpoints
            training_config_path = os.path.join(model_path, "training_config.yaml")
        
        if os.path.exists(training_config_path):
            # Assemble tokenizer from training config
            print(f"Assembling tokenizer from training config at: {training_config_path}")
            tokenizer = assemble_tokenizer(load_config(training_config_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # Assume it's a model name on HuggingFace
        print(f"Loading tokenizer from HuggingFace model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model or directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--annotate_smiles", action="store_true", help="Enable SMILES annotation with [START_SMILES] tags")
    args = parser.parse_args()
    model_path = args.model_path
    output_dir = args.output_dir
    debug = args.debug

    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"Running with {world_size} GPUs")
    
    os.makedirs(output_dir, exist_ok=True)

    load_dotenv()

    # Find all checkpoints to evaluate
    checkpoint_paths = find_checkpoint_folders(model_path)
    
    if rank == 0:
        print(f"Found {len(checkpoint_paths)} checkpoint(s) to evaluate:")
        for cp in checkpoint_paths:
            print(f"  - {cp}")
    
    # Distribute checkpoints across ranks
    # Each rank processes a subset of checkpoints
    my_checkpoints = [cp for i, cp in enumerate(checkpoint_paths) if i % world_size == rank]
    
    print(f"Rank {rank}: Processing {len(my_checkpoints)} checkpoint(s)")
    for cp in my_checkpoints:
        print(f"  Rank {rank}: {cp}")

    # Get tokenizer
    tokenizer = get_tokenizer_for_eval(model_path)

    # Each rank evaluates its assigned checkpoints
    local_results = []
    for checkpoint_path in my_checkpoints:
        print(f"\nRank {rank}: Evaluating checkpoint: {checkpoint_path}")
        eval_results = eval_path(checkpoint_path, tokenizer, local_rank, debug=debug, use_smiles_annotation=args.annotate_smiles)
        log_path_results(eval_results)
        local_results.append(eval_results)

    # Gather results from all ranks
    if world_size > 1:
        # Serialize results to gather them
        local_results_json = json.dumps(local_results)
        
        # Gather all results on rank 0
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results_json)
        
        if rank == 0:
            all_results = []
            for result_json in gathered_results:
                results = json.loads(result_json)
                all_results.extend(results)
            # Sort by checkpoint path to maintain consistent ordering
            all_results.sort(key=lambda x: x['path'])
        else:
            all_results = []
    else:
        all_results = local_results

    # Log summary and save results from rank 0
    if rank == 0:
        log_all_results(all_results)
        filename = model_path.replace("/", "_") + "_results.json"
        with open(f"{output_dir}/{filename}", "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {output_dir}/{filename}")
    
    cleanup_distributed()
