import os
import json
from rdkit import RDLogger

from chembench.prompter import PrompterBuilder
from chembench.types import Generation, Generations
from ChemIQ.utils.answer_verifier import AnswerVerifier
from chembench.utils import enable_caching, enable_logging
from chembench.evaluate import ChemBenchmark, save_topic_reports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# TODO: remove this and check parsing error
RDLogger.DisableLog('rdApp.*')

class ModelWrapper:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def format_prompt(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_text(self, messages):
        formatted_prompt = self.format_prompt(messages)
        generation_args = {
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        output = self.pipeline(formatted_prompt, **generation_args)
        return output[0]["generated_text"]

    def generate(self, prompts, **kwargs):
        generations = []
        for prompt in prompts:
            generation = self.generate_text(prompt)
            generations.append([Generation(text=generation)])

        return Generations(generations=generations)

def run_chembench(model_path, output_path):
    # TODO: make sure this does not create problems when running benchmark on two different models
    enable_caching() 
    enable_logging()
    
    benchmark = ChemBenchmark.from_huggingface()

    model = ModelWrapper(model_path)
    prompter = PrompterBuilder.from_model_object(model=model)

    results = benchmark.bench(prompter)

    os.makedirs(output_path, exist_ok=True)
    save_topic_reports(benchmark, results, output_path)

    # benchmark.submit(results)

def run_chemiq(model_path, chemiq_path, output_path):
    questions = []
    jsonl_path = os.path.join(chemiq_path, "questions", "chemiq.jsonl")
    with open(jsonl_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))

    # TODO: process additional smiles-iupac questions?

    model = ModelWrapper(model_path)
    verifier = AnswerVerifier(jsonl_path)
        
    os.makedirs(output_path, exist_ok=True)
    results_file = os.path.join(output_path, "chemiq_results.jsonl")

    results = []
    for _, question in enumerate(questions):
        id = question.get("uuid")
        prompt = question.get("prompt")
        messages = [{"role": "user", "content": prompt}]

        try:
            try:
                answer = model.generate_text(messages)
            except Exception as e:
                answer = None
                print(f"Failed to generate answer for {id}: {e}")

            record = {
                "uuid": id,
                "question_category": question.get("question_category"),
                "sub_category": question.get("sub_category"),
                "prompt": prompt,
                "expected_answer": question.get("answer"),
                "verification_method": question.get("verification_method"),
                "model_answer": answer,
            }

            if answer is not None:
                try:
                    check = verifier.check_answer(id, answer)
                    record.update(check)
                except Exception as e:
                    print(f"Failed to verify answer for {id}: {e}")
                    record.update({"is_correct": False, "opsin_smiles": None})
            else:
                record.update({"is_correct": None, "opsin_smiles": None})

            results.append(record)
            with open(results_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
        
        except Exception as e:
            fail_record = {
                "uuid": id,
                "question_category": question.get("question_category"),
                "sub_category": question.get("sub_category"),
                "prompt": prompt,
                "expected_answer": question.get("answer"),
                "verification_method": question.get("verification_method"),
                "model_answer": None,
                "is_correct": None,
                "opsin_smiles": None,
            }
            results.append(fail_record)
            with open(results_file, 'a') as f:
                f.write(json.dumps(fail_record) + "\n")

    # TODO: is this all we care about?
    summary = {
        "num_questions": len(questions),
        "num_processed": len(results),
        "num_attempted": sum(1 for r in results if r.get("is_correct") is not None),
        "num_correct": sum(1 for r in results if r.get("is_correct") is True),
        "accuracy": None,
    }

    if summary["num_attempted"] > 0:
        summary["accuracy"] = summary["num_correct"] / summary["num_attempted"]
    else:
        summary["accuracy"] = 0.0

    summary_path = os.path.join(output_path, "chemiq_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
