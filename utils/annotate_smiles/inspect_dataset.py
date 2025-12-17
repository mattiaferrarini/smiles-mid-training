import sys
import logging
import re
import os
import argparse 
import yaml 
from collections import Counter
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from create_smiles import annotate_smiles

def load_config(config_path):
    """
    It loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

load_dotenv()


def main():
    """
    It inspects a dataset of text entries, annotates SMILES strings using regex or an LLM-regex logic,
    it compares the results to annotations in a pre-annotated dataset.
    """

    parser = argparse.ArgumentParser(description="Inspect Dataset Script")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file (e.g., configs/utils/inspect_config.yaml)"
    )
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    USE_LLM = config.get("use_llm", False)
    WHOLE_DATASET = config.get("whole_dataset", True)
    SAVE_NEW_DATASET = config.get("save_new_dataset", False)
    NEW_DATASET_OUTPUT_PATH = config.get("new_dataset_output_path", "processed_dataset_output")
    DATA_FOLDER = config.get("data_folder", "")
    DATA_FILES_PATTERN = config.get("data_files_pattern", "**/*.arrow")
    OUTPUT_FILE = config.get("output_log_file", "dataset_inspection.txt")
    RANDOM_SEED = config.get("random_seed", 42)
    GEMINI_MODEL_NAME = config.get("gemini_model", "gemini-1.5-flash")
    ORIGINAL_TEXT_FIELD = config.get("original_text_field", "text_annotated_v1")
    ORIGINAL_ANNOTATED_FIELD = config.get("original_annotated_field", "text_annotated_v1tags")
    ORIGINAL_SMILES_COUNT_FIELD = config.get("original_smiles_count_field", "text_annotated_v1_smiles_count")
    COMPARE_TO_GT = config.get("compare_to_gt", True)

    # LLM Configuration 
    if USE_LLM:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            print("Error: GEMINI_API_KEY not found in environment variables.")
            USE_LLM = False
        else:
            genai.configure(api_key=api_key)
            # Using 'gemini-1.5-flash' for speed and cost-efficiency
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    if not OUTPUT_FILE:
        OUTPUT_FILE = f"dataset_head_{'whole' if WHOLE_DATASET else 'sample'}_{'llm' if USE_LLM else 'no_llm'}.txt"

    # Configure logging to write to file
    logging.basicConfig(
        filename=OUTPUT_FILE, level=logging.INFO, format="%(message)s", filemode="w", force=True
    )




    print(f"Starting inspection. Output will be saved to {OUTPUT_FILE}")

    try:
        print(f"Loading dataset from {DATA_FOLDER}...")
        

        #print("Dataset loaded (streaming mode). Fetching first 20 rows...")

        RANDOM_SEED = 42
        if not WHOLE_DATASET:
            dataset = load_dataset(
            "arrow",
            data_dir=DATA_FOLDER,
            data_files=DATA_FILES_PATTERN,
            split="train",
            streaming=True,
        )
            shuffled_dataset = dataset.shuffle(seed=RANDOM_SEED)
            random_samples = shuffled_dataset.take(30)
            shuffled_dataset = random_samples
            lenght = 30
        else:
            dataset = load_dataset(
            "arrow",
            data_dir=DATA_FOLDER,
            data_files=DATA_FILES_PATTERN,
            split="train",
        )
            shuffled_dataset = dataset.shuffle(seed=RANDOM_SEED)
            lenght = len(shuffled_dataset)        



        logging.info("--- DATASET INSPECTION ---")
        logging.info(f"Source: {DATA_FOLDER}")
        logging.info(f"Pattern: {DATA_FILES_PATTERN}")

        count = 0
        #all_false_positives = set()
        all_false_positives = Counter() 
        all_false_negatives = Counter()
        total_tp = 0
        total_fp = 0
        total_fn = 0
        print(f"Dataset length: {lenght} rows")
        new_data_buffer = []
        for i, example in enumerate(shuffled_dataset):
            if WHOLE_DATASET and i % (lenght // 100) == 0:
                print(f"Processing row {i+1}/{lenght}...")
                logging.info(f"--- ROW {i+1} ---")

            original_text = example.get(ORIGINAL_TEXT_FIELD, "")
            annotated_text = example.get(ORIGINAL_ANNOTATED_FIELD, "")
            annotated_smiles_count = example.get(ORIGINAL_SMILES_COUNT_FIELD, "N/A")

            new_text, my_smiles_count = annotate_smiles(original_text, USE_LLM=USE_LLM, model=model if USE_LLM else None)


            if SAVE_NEW_DATASET:
                # We create a copy of the row
                new_row = example.copy()
                # We add/overwrite the text with the NEW annotated version
                # You can overwrite 'text_annotated_v1tags' or create a new column
                new_row['text_reannotated'] = new_text 
                new_row['reannotated_smiles_count'] = my_smiles_count
                new_data_buffer.append(new_row)

            # Extraction and Comparison
            my_smiles = set(s.strip() for s in re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", new_text))

            if COMPARE_TO_GT:
                gt_smiles = set(s.strip() for s in re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", annotated_text))

                tp = len(gt_smiles.intersection(my_smiles))
                fp_set = my_smiles - gt_smiles
                fn_set = gt_smiles - my_smiles
                fp = len(fp_set)
                fn = len(fn_set)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn


                all_false_positives.update(fp_set)
                all_false_negatives.update(fn_set)

            if not WHOLE_DATASET:
                logging.info(f"--- ROW {i+1} ---")
                if COMPARE_TO_GT:
                    logging.info(f"Original Smiles Count (Variable): {annotated_smiles_count}")
                logging.info(f"My Regex Smiles Count: {my_smiles_count}")
                if COMPARE_TO_GT:
                    logging.info(f"False Positives (My Regex found, but not in GT): {list(fp_set)}")
                    logging.info(f"False Negatives (In GT, but My Regex missed): {list(fn_set)}")
                    
                    logging.info("Confusion Matrix:")
                    logging.info(f"\t\tPred +\tPred -")
                    logging.info(f"Act +\t{tp}\t{fn}")
                    logging.info(f"Act -\t{fp}\tX")

                    logging.info("-" * 20)
                    logging.info("Original Text Annotated V1:")
                    logging.info(original_text)
                    logging.info("-" * 20)
                else:
                    logging.info("Original Text")
                    logging.info(original_text)
                logging.info("New Version (with [START_SMILES] tags):")
                logging.info(new_text)
                logging.info("\n")
                logging.info("=" * 50)
                logging.info("\n")

            count += 1

        if count == 0:
            logging.warning("Dataset seems empty.")


        if SAVE_NEW_DATASET and count > 0:
            print(f"Saving new dataset with {len(new_data_buffer)} rows to {NEW_DATASET_OUTPUT_PATH}...")
            try:
                # Create a Hugging Face Dataset object from the list of dicts
                new_dataset = Dataset.from_list(new_data_buffer)
                new_dataset.save_to_disk(NEW_DATASET_OUTPUT_PATH)
                print("Dataset saved successfully.")
            except Exception as e:
                print(f"FAILED to save dataset: {e}")
                logging.error(f"FAILED to save dataset: {e}")

        if COMPARE_TO_GT:
            logging.info("=" * 50)
            logging.info("TOP 1000 MOST FREQUENT FALSE POSITIVES (Predicted but NOT in GT):")
            logging.info("Count\tString")
            # most_common returns a list of (element, count) tuples
            for smi, freq in all_false_positives.most_common(1000):
                logging.info(f"{freq}\t{smi}")
            
            # Added logic to print False Negatives sorted by frequency
            logging.info("=" * 50)
            logging.info("TOP 1000 MOST FREQUENT FALSE NEGATIVES (In GT but MISSED by Regex):")
            logging.info("Count\tString")
            for smi, freq in all_false_negatives.most_common(1000):
                logging.info(f"{freq}\t{smi}")


            logging.info("=" * 50)

            logging.info("FINAL AGGREGATED CONFUSION MATRIX:")
            logging.info(f"\t\tPred +\tPred -")
            logging.info(f"Act +\t{total_tp}\t{total_fn}")
            logging.info(f"Act -\t{total_fp}\tX")
            logging.info("=" * 50)

        print(f"Done. {count} rows written to {OUTPUT_FILE}")

    except Exception as e:
        error_msg = f"Error inspecting dataset: {e}"
        print(error_msg)
        logging.error(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()