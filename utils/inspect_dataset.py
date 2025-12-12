import sys
import logging
from datasets import load_dataset
from utils.create_smiles import annotate_smiles

# Hardcoded variables
DATA_FOLDER = "/capstor/store/cscs/swissai/a131/ML4Science/datasets/CEMB_v1_HF_FineWeb-chemV1"
DATA_FILES_PATTERN = "**/*.arrow"
OUTPUT_FILE = "dataset_head.txt"

# Configure logging to write to file
logging.basicConfig(
    filename=OUTPUT_FILE,
    level=logging.INFO,
    format='%(message)s',
    filemode='w'
)


print(f"Starting inspection. Output will be saved to {OUTPUT_FILE}")

try:
    print(f"Loading dataset from {DATA_FOLDER}...")
    dataset = load_dataset(
        "arrow", 
        data_dir=DATA_FOLDER, 
        data_files=DATA_FILES_PATTERN, 
        split="train",
        streaming=True 
    )
    
    print("Dataset loaded (streaming mode). Fetching first 20 rows...")
    
    # taking 20 random samples
    RANDOM_SEED = 42
    shuffled_dataset = dataset.shuffle(seed=RANDOM_SEED)
    random_samples = shuffled_dataset.take(20)
    
    logging.info("--- DATASET INSPECTION ---")
    logging.info(f"Source: {DATA_FOLDER}")
    logging.info(f"Pattern: {DATA_FILES_PATTERN}")

    count = 0
    for i, example in enumerate(random_samples):
        logging.info(f"--- ROW {i+1} ---")
        
        original_text = example.get("text_annotated_v1", "")
        annotated_smiles_count = example.get("text_annotated_v1_smiles_count", "N/A")
        
        new_text, my_smiles_count = annotate_smiles(original_text)
        
        logging.info(f"Original Smiles Count (Variable): {annotated_smiles_count}")
        logging.info(f"My Regex Smiles Count: {my_smiles_count}")
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
        
    print(f"Done. {count} rows written to {OUTPUT_FILE}")

except Exception as e:
    error_msg = f"Error inspecting dataset: {e}"
    print(error_msg)
    logging.error(error_msg)
    sys.exit(1)