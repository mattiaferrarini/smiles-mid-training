import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse
import json
import os

DATASET = "jablonkagroup/ChemBench"

configs = ['analytical_chemistry', 'chemical_preference', 'general_chemistry', 
           'inorganic_chemistry', 'materials_science', 'organic_chemistry', 
           'physical_chemistry', 'technical_chemistry', 'toxicity_and_safety']


def setup_distributed():
    """Initialize distributed training environment"""
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


def get_likelihood(model, tokenizer, question, answer, debug=False):
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


def eval_config(model, tokenizer, config, rank, world_size, debug=False):
    # Filter for multiple choice questions
    ds = load_dataset(DATASET, config)["train"]
    ds = ds.filter(lambda x: x["metrics"] == ["multiple_choice_grade"])
    
    if rank == 0:
        print(f"Evaluating {config}: {len(ds)} questions.")
    
    # Split dataset across GPUs
    # Each rank processes a subset of the dataset
    total_examples = len(ds)
    examples_per_rank = total_examples // world_size
    start_idx = rank * examples_per_rank
    end_idx = start_idx + examples_per_rank if rank < world_size - 1 else total_examples
    
    if rank == 0:
        print(f"Total examples: {total_examples}, each rank processes ~{examples_per_rank} examples")
    
    correct_count = 0
    total_count = 0
    
    # Only process the subset assigned to this rank
    for idx in tqdm(range(start_idx, end_idx), desc=f"Rank {rank}", disable=(rank != 0)):
        row = ds[idx]
        examples_list = row.get("examples")
        
        if not examples_list:
            continue
            
        for example in examples_list:
            question, target_scores, candidates = process_example(example)
            if question is None or target_scores is None or candidates is None:
                continue

            candidate_scores = []

            if debug and rank == 0:
                print(f"Question: {question}")
            # Calculate likelihoods
            for candidate in candidates:
                score = get_likelihood(model, tokenizer, question, candidate, debug=debug)
                if debug and rank == 0:
                    print(f"Candidate: {candidate}, Score: {score}")
                candidate_scores.append(score)

            # Select best candidate
            if debug and rank == 0:
                print("Candidate scores:", candidate_scores)
                print("Target scores:", target_scores)

            best_idx = np.argmax(candidate_scores)
            predicted_answer = candidates[best_idx]
            if debug and rank == 0:
                print(f"Predicted answer: {predicted_answer}")

            # Check correctness
            if target_scores[predicted_answer] == 1:
                correct_count += 1
            total_count += 1

    # Gather results from all ranks
    if world_size > 1:
        correct_tensor = torch.tensor([correct_count], dtype=torch.long, device=torch.cuda.current_device())
        total_tensor = torch.tensor([total_count], dtype=torch.long, device=torch.cuda.current_device())
        
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        correct_count = correct_tensor.item()
        total_count = total_tensor.item()

    accuracy = correct_count / total_count if total_count > 0 else 0
    if rank == 0:
        print(f"Accuracy for {config}: {accuracy:.2%} ({correct_count}/{total_count})")
    return {"config": config, "accuracy": accuracy, "correct": correct_count, "total": total_count}


def evaluate(model, tokenizer, rank, world_size, debug=False):
    results = []
    for config in configs:
        result = eval_config(model, tokenizer, config, rank, world_size, debug=debug)
        results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
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

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model on the correct device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and world_size > 1 else "cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    
    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    model.eval()
    results = evaluate(model, tokenizer, rank, world_size, debug=debug)

    # Only save results from rank 0
    if rank == 0:
        filename = model_path.replace("/", "_") + "_results.json"
        with open(f"{output_dir}/{filename}", "w") as f:
            json.dump(results, f, indent=4)
        print("Results saved to", f"{output_dir}/{filename}")
    
    cleanup_distributed()