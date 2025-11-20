from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils.logging import get_logger
from data.loading import load_train_eval_datasets
from data.instruction import load_instruction_datasets
from embeddings.registry import build_embedding_strategy
from embeddings.utils import resolve_embedding_config, summarise_embedding_sections

LOGGER = get_logger(__name__)

def _default_training_section():
    return {
        "output_dir": "artifacts/runs/default",
        "epochs": 1,
        "num_train_epochs": 1,
        "batch_size": 2,
        "per_device_train_batch_size": 2,
        "eval_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": 5.0e-5,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "logging_steps": 10,
        "evaluation_strategy": "no",
        "save_strategy": "no",
        "seed": 42,
    }

#wandb update#
import subprocess
import shutil
import logging

def _normalise_training_config(raw_cfg):
    cfg = {**_default_training_section(), **raw_cfg}

    cfg["num_train_epochs"] = cfg.get("num_train_epochs", cfg.get("epochs", 1))
    cfg["per_device_train_batch_size"] = cfg.get(
        "per_device_train_batch_size", cfg.get("batch_size", 2)
    )
    cfg["per_device_eval_batch_size"] = cfg.get(
        "per_device_eval_batch_size",
        cfg.get("eval_batch_size", cfg.get("batch_size", 2)),
    )

    cfg["epochs"] = cfg["num_train_epochs"]
    cfg["batch_size"] = cfg["per_device_train_batch_size"]
    cfg["eval_batch_size"] = cfg["per_device_eval_batch_size"]

    return cfg


def build_training_arguments(config, section="training"):
    raw_cfg = config.get(section, {})
    training_cfg = _normalise_training_config(raw_cfg)

    return TrainingArguments(
        output_dir=str(training_cfg["output_dir"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        warmup_steps=int(training_cfg.get("warmup_steps", 0)),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "no"),
        save_strategy=training_cfg.get("save_strategy", "no"),
        seed=int(training_cfg.get("seed", 42)),
        report_to=[],
        remove_unused_columns=False,
    )


def prepare_model_and_tokenizer(
    config,
    tokenizer,
    hf_token,
):
    model_name = config["model"]["name"]
    LOGGER.info("Loading model %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    original_vocab_size = model.get_input_embeddings().weight.size(0)

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, original_vocab_size


def run_mixed_domain_training(
    model,
    tokenizer,
    config,
):
    try:
        train_dataset, eval_dataset, train_stats, eval_stats = load_train_eval_datasets(
            config,
            tokenizer,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Skipping mixed-domain training: %s", exc)
        return False

    LOGGER.info(
        "Training samples -> total: %.0f, chemical: %.0f, general: %.0f",
        train_stats["total_samples"],
        train_stats["chemical_samples"],
        train_stats["general_samples"],
    )
    LOGGER.info(
        "Eval samples -> total: %.0f, chemical: %.0f, general: %.0f",
        eval_stats["total_samples"],
        eval_stats["chemical_samples"],
        eval_stats["general_samples"],
    )

    training_args = build_training_arguments(config, section="training")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    if training_args.save_strategy != "no":
        trainer.save_model()
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(training_args.output_dir)

    return True


def run_instruction_tuning(
    model,
    tokenizer,
    config,
):
    instruction_cfg = config.get("instruction_training", {})
    if not instruction_cfg.get("enabled", False):
        LOGGER.info("Instruction tuning disabled; skipping")
        return False

    try:
        train_dataset, eval_dataset, train_stats, eval_stats = load_instruction_datasets(
            config,
            tokenizer,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Skipping instruction tuning: %s", exc)
        return False

    LOGGER.info("Instruction train samples -> total: %d", train_stats["total_samples"])
    LOGGER.info("Instruction eval samples -> total: %d", eval_stats["total_samples"])

    training_args = build_training_arguments(config, section="instruction_training")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    if training_args.save_strategy != "no":
        trainer.save_model()
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(training_args.output_dir)

    return True


def resize_embeddings_if_needed(
    model,
    tokenizer,
    embedding_cfg,
):
    original_vocab_size = model.get_input_embeddings().weight.size(0)
    target_vocab_size = len(tokenizer)

    if target_vocab_size <= original_vocab_size:
        LOGGER.info("No embedding resize required (vocab unchanged)")
        return

    model.resize_token_embeddings(target_vocab_size)
    num_new_tokens = target_vocab_size - original_vocab_size
    strategy = build_embedding_strategy(embedding_cfg)
    strategy(model, num_new_tokens)
    weight = model.get_input_embeddings().weight.detach()
    stats = summarise_embedding_sections(weight, num_new_tokens)

    LOGGER.info(
        "Resized embeddings from %d to %d", original_vocab_size, target_vocab_size
    )
    LOGGER.info(
        "New embedding stats %s",
        ", ".join(f"{k}={v:.6f}" for k, v in stats.items() if k.startswith("new_")),
    )


def seed_everything(config):
    training_cfg = config.get("training") or {}
    seed_value = training_cfg.get("seed")
    if seed_value is None:
        seed_value = _default_training_section()["seed"]
    if seed_value is not None:
        set_seed(int(seed_value))


def prepare_tokenizer(config, hf_token):
    from tokenizer.registry import build_tokenizer_from_config

    tokenizer_cfg = config.get("tokenizer") or {}
    tokenizer_type = tokenizer_cfg.get("type", "element")
    LOGGER.info("Loading tokenizer type '%s'", tokenizer_type)
    tokenizer = build_tokenizer_from_config(config, hf_token)
    return tokenizer


def run_training_pipeline(config, hf_token, embedding_override=None):
    seed_everything(config)
    tokenizer = prepare_tokenizer(config, hf_token)
    embedding_cfg = resolve_embedding_config(config.get("embeddings"), embedding_override)
    LOGGER.info("Using embedding strategy '%s'", embedding_cfg["strategy"])

    model, _ = prepare_model_and_tokenizer(config, tokenizer, hf_token)
    resize_embeddings_if_needed(model, tokenizer, embedding_cfg)

    mixed_ok = run_mixed_domain_training(model, tokenizer, config)
    if not mixed_ok:
        LOGGER.warning("Mixed-domain stage did not complete successfully")

    run_instruction_tuning(model, tokenizer, config)
