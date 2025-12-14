import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import csv
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dotenv import load_dotenv
from utils.logging import get_logger
from custom_tokenizers.assemble_tokenizer import assemble_tokenizer
from utils.config import load_config

LOGGER = get_logger(__name__)


def load_data(dataset_path, smiles_col, label_col):
    """
    Loads a dataset from a CSV file and extracts SMILES strings and labels.
    Args:
        dataset_path (str): Path to the CSV dataset file.
        smiles_col (str): Name of the column containing SMILES strings.
        label_col (str): Name of the column containing labels.
    Returns:
        tuple: A tuple containing a list of SMILES strings and a list of labels.
    """
    dataset = pd.read_csv(dataset_path)
    smiles_list = dataset[smiles_col].tolist()
    labels = dataset[label_col].tolist()
    return smiles_list, labels


def compute_embeddings(tokenizer, model, smiles_list, batch_size=32):
    """
    Computes embeddings for a list of SMILES strings using the provided tokenizer and model.
    Args:
        tokenizer: The tokenizer to use for encoding SMILES strings.
        model: The model to get the embeddings from using the embedding matrix.
        smiles_list (list): A list of SMILES strings.
        batch_size (int): The batch size for processing SMILES strings.
    Returns:
        list: A list of embeddings.
    """
    embeddings = []
    device = model.device

    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Computing embeddings"):
        batch_smiles = smiles_list[i:i + batch_size]

        # Ensure tokenizer has a pad token        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # If no eos_token, add a pad token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})


        # Tokenize the batch
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            # Get embeddings from the model's embedding matrix
            embedding_layer = model.get_input_embeddings()
            token_embeddings = embedding_layer(input_ids)
            
            # Average embeddings, ignoring padding
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            batch_embeddings = sum_embeddings / sum_mask
            
            embeddings.extend(batch_embeddings.cpu().tolist())
            
    return embeddings


def plot_and_save_embeddings(embeddings, labels, output_path, plot_title, topk=10):
    """
    Plots embeddings using t-SNE and saves the plot to a file (SVG format).
    Args:
        embeddings (list): A list of embeddings.
        labels (list): A list of labels corresponding to the embeddings.
        output_path (str): Path to save the plot image (will be saved as SVG).
    """

    embeddings_arr = np.array(embeddings)
    labels_arr = np.array(labels)

    # Exclude missing labels so they don't appear as a class in the plot
    non_na_mask = ~pd.isna(labels_arr)
    if not non_na_mask.all():
        LOGGER.info(f"Excluding {np.sum(~non_na_mask)} samples with missing labels.")
    embeddings_arr = embeddings_arr[non_na_mask]
    labels_arr = labels_arr[non_na_mask]

    most_common = Counter(labels_arr).most_common(topk)
    top_classes = [label for label, count in most_common]
    print(f"Plotting the following classes: {top_classes}")
    mask = np.isin(labels_arr, top_classes)
    
    filtered_embeddings = embeddings_arr[mask]
    filtered_labels = labels_arr[mask]

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(filtered_embeddings)

    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, topk))
    for i, label in enumerate(top_classes):
        idx = (filtered_labels == label)
        plt.scatter(
            reduced_embeddings[idx, 0], 
            reduced_embeddings[idx, 1], 
            color=colors[i], 
            label=str(label),
            alpha=0.7,
            s=5
        )

    output_path_svg = output_path if output_path.endswith('.svg') else output_path + '.svg'
    output_path_png = output_path if output_path.endswith('.png') else output_path + '.png'
    plt.legend(title=f"Classes")
    plt.title(plot_title)
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
    plt.savefig(output_path_png, format='png', bbox_inches='tight')
    plt.close()


def get_tokenizer(checkpoint_folder):
    """
    Loads the appropriate tokenizer for evaluation based on the model path

    It first checks for training_config.yaml to assemble a custom tokenizer
    If not found, it falls back to loading the tokenizer directly from the model path using AutoTokenizer

    Args:
        model_path (str): Path to the pretrained model or directory containing checkpoints

    Returns:
        PreTrainedTokenizer: The loaded tokenizer instance
    """
    if os.path.exists(checkpoint_folder) and os.path.isdir(checkpoint_folder):
        # Local directory
        if os.path.exists(os.path.join(checkpoint_folder, "config.json")):
            # Single checkpoint folder: find config in parent
            training_config_path = os.path.join(
                os.path.dirname(checkpoint_folder), "training_config.yaml"
            )
        else:
            # Folder with multiple checkpoints
            training_config_path = os.path.join(checkpoint_folder, "training_config.yaml")

        if os.path.exists(training_config_path):
            # Assemble tokenizer from training config
            print(
                f"Assembling tokenizer from training config at: {training_config_path}"
            )
            tokenizer = assemble_tokenizer(load_config(training_config_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
    else:
        # Assume it's a model name on HuggingFace
        print(f"Loading tokenizer from HuggingFace model: {checkpoint_folder}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)

    return tokenizer


def evaluate_embeddings(checkpoint_folder, dataset_path, smiles_col, label_col, output_path, plot_title):
    """
    Evaluates embeddings of a model checkpoint on a dataset and plots the results.
    Args:
        checkpoint_folder (str): Path to the model checkpoint folder.
        dataset_path (str): Path to the dataset CSV file.
        smiles_col (str): Name of the column containing SMILES strings.
        label_col (str): Name of the column containing labels.
        output_path (str): Path to save the embeddings plot.
    Returns:
        None
    """

    load_dotenv()
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    tokenizer = get_tokenizer(checkpoint_folder)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_folder)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    smiles_list, labels = load_data(dataset_path, smiles_col, label_col)
    LOGGER.info(f"Loaded {len(smiles_list)} SMILES from dataset.")
    embeddings = compute_embeddings(tokenizer, model, smiles_list, batch_size=16)
    plot_and_save_embeddings(embeddings, labels, output_path, plot_title, topk=10)
    