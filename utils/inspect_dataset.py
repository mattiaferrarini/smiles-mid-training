import sys
import logging
import re
from datasets import load_dataset
from create_smiles import annotate_smiles
from dotenv import load_dotenv
import os

load_dotenv()
# this is just in case it is necessary to be extremely precise. Choose the model accordingly for your use case
USE_LLM = False

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
        model = genai.GenerativeModel('gemini-1.5-flash')

whole_dataset = True
# Hardcoded variables
DATA_FOLDER = (
    "/capstor/store/cscs/swissai/a131/ML4Science/datasets/CEMB_v1tags_HF_FineWeb-chemV1"
)
DATA_FILES_PATTERN = "**/*.arrow"
OUTPUT_FILE = f"dataset_head_{'whole' if whole_dataset else 'sample'}_{'llm' if USE_LLM else 'no_llm'}.txt"

# Configure logging to write to file
logging.basicConfig(
    filename=OUTPUT_FILE, level=logging.INFO, format="%(message)s", filemode="w"
)




print(f"Starting inspection. Output will be saved to {OUTPUT_FILE}")

try:
    print(f"Loading dataset from {DATA_FOLDER}...")
    

    #print("Dataset loaded (streaming mode). Fetching first 20 rows...")

    RANDOM_SEED = 42
    if not whole_dataset:
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
    all_false_positives = set()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    print(f"Dataset length: {lenght} rows")
    for i, example in enumerate(shuffled_dataset):
        if whole_dataset and i % (lenght // 100) == 0:
            print(f"Processing row {i+1}/{lenght}...")
            logging.info(f"--- ROW {i+1} ---")

        original_text = example.get("text_annotated_v1", "")
        annotated_text = example.get("text_annotated_v1tags", "")
        annotated_smiles_count = example.get("text_annotated_v1_smiles_count", "N/A")

        new_text, my_smiles_count = annotate_smiles(original_text, USE_LLM=USE_LLM, model=model if USE_LLM else None)

        # Extraction and Comparison
        gt_smiles = set(s.strip() for s in re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", annotated_text))
        my_smiles = set(s.strip() for s in re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", new_text))

        tp = len(gt_smiles.intersection(my_smiles))
        fp_set = my_smiles - gt_smiles
        fn_set = gt_smiles - my_smiles
        fp = len(fp_set)
        fn = len(fn_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_false_positives.update(fp_set)

        if not whole_dataset:
            logging.info(f"--- ROW {i+1} ---")
            logging.info(f"Original Smiles Count (Variable): {annotated_smiles_count}")
            logging.info(f"My Regex Smiles Count: {my_smiles_count}")
            
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
            logging.info("My New Version (with [START_SMILES] tags):")
            logging.info(new_text)
            logging.info("\n")
            logging.info("=" * 50)
            logging.info("\n")

        count += 1

    if count == 0:
        logging.warning("Dataset seems empty.")

    logging.info("=" * 50)
    logging.info("ALL FALSE POSITIVES FOUND ACROSS ALL SAMPLES:")
    if not whole_dataset:
        for fp_item in sorted(list(all_false_positives)):
            logging.info(fp_item)
    else:
        most_recurrent_false_positives = sorted(list(all_false_positives))[:1000]
        for fp_item in most_recurrent_false_positives:
            logging.info(fp_item)
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
