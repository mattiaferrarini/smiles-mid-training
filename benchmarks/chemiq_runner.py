import sys
import json
import typer
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add ChemIQ to path to allow imports
sys.path.append("ChemIQ")
try:
    from utils.parser import AnswerParser
    from utils.answer_verifier import AnswerVerifier
except ImportError:
    print("Error: Could not import ChemIQ utils. Make sure the ChemIQ folder is in the current directory.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("chemiq_runner")

app = typer.Typer()

def load_questions(question_file: Path):
    with open(question_file, "r") as f:
        questions = [json.loads(line) for line in f]
    # Filter for ChemIQ benchmark questions
    return [q for q in questions if q.get("ChemIQ", False)]

@app.command()
def main(
    model_path: Path = typer.Option(..., "--model-path", help="Path to the model directory"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to save results"),
    question_file: Path = typer.Option(Path("ChemIQ/questions/chemiq.jsonl"), "--question-file", help="Path to chemiq.jsonl"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size for generation"),
    device: str = typer.Option("auto", "--device", help="Device to use (cuda/cpu/auto)"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Questions
    LOGGER.info(f"Loading questions from {question_file}")
    questions = load_questions(question_file)
    LOGGER.info(f"Found {len(questions)} ChemIQ questions")

    # 2. Initialize Helpers
    parser = AnswerParser(str(question_file))
    verifier = AnswerVerifier(str(question_file))

    # 3. Load Model
    LOGGER.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True
    )
    model.eval()

    # 4. Run Inference
    results = []
    correct_count = 0

    for q in tqdm(questions, desc="Evaluating"):
        uuid = q["uuid"]
        prompt_text = q["prompt"]
        
        # Format prompt using chat template if available
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for base models
            prompt = prompt_text

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # Greedy decoding for reproducibility
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the new tokens
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 5. Parse and Verify
        try:
            parsed_answer = parser.parse(uuid, generated_text)
            verification = verifier.check_answer(uuid, parsed_answer)
            is_correct = verification.get("is_correct", False)
        except Exception as e:
            LOGGER.error(f"Error processing {uuid}: {e}")
            parsed_answer = None
            is_correct = False
            verification = {"error": str(e)}

        if is_correct:
            correct_count += 1

        results.append({
            "uuid": uuid,
            "category": q["question_category"],
            "prompt": prompt_text,
            "generated_text": generated_text,
            "parsed_answer": str(parsed_answer),
            "is_correct": is_correct,
            "details": verification
        })

    # 6. Save Results
    accuracy = correct_count / len(questions) if questions else 0.0
    LOGGER.info(f"Evaluation complete. Accuracy: {accuracy:.2%}")

    summary = {
        "model": str(model_path),
        "accuracy": accuracy,
        "total": len(questions),
        "correct": correct_count,
        "results": results
    }

    output_file = output_dir / "chemiq_results.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    LOGGER.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    app()
