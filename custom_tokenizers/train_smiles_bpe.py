from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import load_dataset
import os
import argparse
import re

def train_smiles_bpe(data_folder, data_files_pattern, text_field, output_dir, vocab_size=1000, min_frequency=2):
    """
    Trains a BPE tokenizer specifically on the SMILES column of a dataset.
    It extracts content between [START_SMILES] and [END_SMILES] tags to ensure
    the tokenizer learns only chemical patterns.
    """
    print(f"Loading dataset from {data_folder} with pattern {data_files_pattern}...")
    
    # Load dataset (matching training_trl.py logic: arrow format, no streaming)
    dataset = load_dataset(
        "arrow", 
        data_dir=data_folder, 
        data_files=data_files_pattern, 
        split="train"
    ).select_columns([text_field])

    # Rename text column if necessary
    if text_field != "text":
        dataset = dataset.rename_column(text_field, "text")
    
    print(f"Dataset loaded. Size: {len(dataset)} rows.")

    # Iterator to yield SMILES strings
    def batch_iterator(batch_size=10000):
        batch = []
        smiles_pattern = re.compile(r"\[START_SMILES\](.*?)\[END_SMILES\]")
        
        # Iterate over the dataset
        for i in range(0, len(dataset), batch_size):
            # Get a chunk of text
            chunk = dataset[i : i + batch_size]["text"]
            for text in chunk:
                if text:
                    # Extract all SMILES segments from the text
                    matches = smiles_pattern.findall(text)
                    for match in matches:
                        # Add the extracted SMILES to the batch
                        batch.append(match.strip())
            
            # Yield if we have enough
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining
        if batch:
            yield batch

    # Initialize BPE Tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenization: Whitespace is safe for now
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[START_SMILES]", "[END_SMILES]"]
    )

    print("Starting training...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    print(f"Tokenizer saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--data_files_pattern", type=str, default="**/*.arrow", help="Pattern for data files")
    parser.add_argument("--text_field", type=str, default="text", help="Column name for text")
    parser.add_argument("--output_dir", type=str, default="custom_tokenizers/smiles_bpe", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=2000, help="Vocab size")
    args = parser.parse_args()

    train_smiles_bpe(args.data_folder, args.data_files_pattern, args.text_field, args.output_dir, args.vocab_size)
