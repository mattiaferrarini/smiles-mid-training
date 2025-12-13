import json
from pathlib import Path
from utils.logging import get_logger

LOGGER = get_logger(__name__)


def _load_vocab_from_json(path, append_to_existing_vocabulary=False, self_vocab=None):
    p = Path(path)
    if not p.exists():
        LOGGER.warning(f"Vocabulary file does not exist at path: {path}")
        return {}
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # normalize types
    vocab = {str(k): int(v) for k, v in raw.items()}
    if append_to_existing_vocabulary:
        for token in self_vocab.keys():
            if token not in vocab:
                vocab[token] = self_vocab[token]
        return vocab
    else:
        # Ensure "[UNK]": 0 exists; if 0 is taken shift existing ids +1
        if "[UNK]" in vocab:
            if vocab["[UNK]"] != 0:
                # shift all ids by +1 and set [UNK]=0
                vocab = {k: (v + 1) if k != "[UNK]" else 0 for k, v in vocab.items()}
                vocab["[UNK]"] = 0
        else:
            if 0 in vocab.values():
                vocab = {k: (v + 1) for k, v in vocab.items()}
            vocab["[UNK]"] = 0
        return vocab


def reset_vocabulary():
    return {"[UNK]": 0}


def load_vocabulary(vocab_path="../json/vocab_symbol_to_number.json"):
    try:
        vocab = _load_vocab_from_json(vocab_path)
        return vocab
    except Exception as e:
        LOGGER.warning(
            f"Json file not found or could not be loaded: {e}", exc_info=True
        )
        return reset_vocabulary()


def create_vocabulary(
    text,
    _tokenize,
    append_to_existing_vocabulary=False,
    vocab_path="../json/vocab_symbol_to_number.json",
    save_vocabulary=False,
    vocab={"[UNK]": 0},
):
    if append_to_existing_vocabulary:
        try:
            vocab_file = _load_vocab_from_json(vocab_path)
            for token in vocab_file.keys():
                if token not in vocab:
                    vocab[token] = vocab_file[token]
        except Exception as e:
            LOGGER.warning(
                f"Json file not found or could not be loaded: {e}", exc_info=True
            )

    # Handle both single string and list of strings
    if isinstance(text, str):
        text_iter = [text]
    else:
        text_iter = text

    max_id = max(vocab.values()) if vocab else -1
    next_id = max_id + 1

    for chunk in text_iter:
        tokens = _tokenize(chunk)
        for token in tokens:
            if token not in vocab:
                vocab[token] = next_id
                next_id += 1

    return vocab


def save_tokenizer_files(save_directory, vocab_dict, config_dict):

    save_directory.mkdir(parents=True, exist_ok=True)

    vocab_file = save_directory / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, indent=4)

    config_file = save_directory / "tokenizer_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    return str(vocab_file), str(config_file)


def build_and_save_tokenizer(
    TokenizerClass,
    dataset,
    text_field,  # text field in the dataset
    output_dir,  # output directory to save the tokenizer
    config=None,  # optional config dict to pass to tokenizer
):
    """
    It learns the vocabulary of the tokenizer and saves it compatible with Huggingface.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tk = TokenizerClass(config=config)
    LOGGER.info("Preparing text corpus...")
    # Pass the list of strings directly, do not join them
    # This prevents learning patterns across SMILES boundaries (e.g. "][")
    all_text = dataset[text_field]

    LOGGER.info(f"Starting vocabulary creation for {TokenizerClass.__name__}...")

    tk.create_vocabulary(all_text, save_vocabulary=False)
    LOGGER.info(f"Vocabulary created with {tk.vocab_size} symbols")

    tk.save_pretrained(output_path)
    LOGGER.info(f"Tokenizer saved in: {output_path.resolve()}")

    return tk
