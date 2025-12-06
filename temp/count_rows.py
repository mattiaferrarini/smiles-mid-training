import sys
import os
import logging
import traceback
from datasets import load_dataset

# Configure logging to write to file instead of console
LOG_FILE = "dataset_count.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(message)s',
    filemode='w'
)

DATA_FOLDER = "/capstor/store/cscs/swissai/a131/ML4Science/datasets/CEMB_v1tags_HF_FineWeb-chemV1"
DATA_FILES_PATTERN = "**/*.arrow"

logging.info("--- Avvio del conteggio righe dataset ---")
logging.info(f"Cartella dati: {DATA_FOLDER}")
logging.info(f"Pattern file: {DATA_FILES_PATTERN}")

# Force cache dir to scratch to avoid Disk Quota Exceeded
user = os.environ.get('USER')
if user:
    cache_dir = f"/iopsstor/scratch/cscs/{user}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
else:
    cache_dir = None

try:
    logging.info("Dataset loading...")
    dataset = load_dataset(
        "arrow", 
        data_dir=DATA_FOLDER, 
        data_files=DATA_FILES_PATTERN, 
        split="train",
        cache_dir=cache_dir
    )
    logging.info("Dataset loaded.")

    row_count = len(dataset)
    logging.info(f"\nConteggio Riuscito!")
    logging.info(f"Total rows in 'train' split: {row_count}")

    # Print first 3 examples
    logging.info("\n--- First 3 examples (text field) ---")
    for i in range(min(3, len(dataset))):
        text = dataset[i].get("text_annotated_v1tags", "[No 'text' field found]")
        # Print the first 500 characters of the text field
        logging.info(f"Row {i} text:\n{text[:500]}...\n")

    logging.info(f"--- Fine del conteggio ---")

except Exception as e:
    error_msg = f"\nCRITICAL ERROR: {e}"
    logging.error(error_msg)
    
    # FORCE PRINT TO STDOUT (so you see it in the Slurm log)
    print(error_msg)
    traceback.print_exc()
    sys.exit(1)