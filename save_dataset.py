import argparse
import os
from datasets import load_dataset, DatasetDict

# Constants
DATASET_NAME = "jablonkagroup/ChemBench"
CONFIGS = [
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

def save_filtered_dataset(output_path):
    """
    Downloads ChemBench, filters for multiple choice questions, 
    and saves to disk as a DatasetDict.
    """
    print(f"Source: {DATASET_NAME}")
    print(f"Destination: {output_path}")
    print("-" * 40)

    collected_datasets = {}

    for config in CONFIGS:
        print(f"Processing {config}...", end=" ", flush=True)
        
        try:
            # Load the specific config (split='train' is standard for this dataset)
            ds = load_dataset(DATASET_NAME, config, split="train")
            original_len = len(ds)

            # Filter logic: Keep only rows where metrics is strictly ["multiple_choice_grade"]
            ds_filtered = ds.filter(lambda x: x["metrics"] == ["multiple_choice_grade"])
            filtered_len = len(ds_filtered)

            collected_datasets[config] = ds_filtered
            print(f"Done. (Kept {filtered_len}/{original_len})")

        except Exception as e:
            print(f"\nError processing {config}: {e}")

    # Combine all configs into a single DatasetDict
    # This preserves the structure: {'organic_chemistry': Dataset(...), ...}
    final_dd = DatasetDict(collected_datasets)

    print("-" * 40)
    print("Saving to disk...")
    final_dd.save_to_disk(output_path)
    print(f"Successfully saved filtered dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save filtered ChemBench dataset.")
    parser.add_argument("output_path", type=str, help="Path to save the dataset (e.g., ./chembench_local)")
    
    args = parser.parse_args()
    
    # Create directory if it doesn't exist (though save_to_disk handles this usually)
    if not os.path.exists(os.path.dirname(args.output_path)) and os.path.dirname(args.output_path) != "":
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    save_filtered_dataset(args.output_path)
