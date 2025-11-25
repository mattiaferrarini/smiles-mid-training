import os
import sys
from pathlib import Path

# Add parent directory to Python path to enable imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset, interleave_datasets
import datasets
from dotenv import load_dotenv
import wandb
import torch
import time
from utils.logging import get_logger

LOGGER = get_logger(__name__)

class MultiGPUResourcesCallback(TrainerCallback):
    def __init__(self, log_steps):
        super().__init__()
        self.log_steps = log_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Print memory usage for every rank to ensure load balancing
        if torch.cuda.is_available():
            rank = int(os.environ.get("RANK", 0))
            if state.global_step % self.log_steps == 0: 
                current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"[Step {state.global_step}] Rank {rank}: {current_mem:.2f} GB (Max: {max_mem:.2f} GB)")


class ThroughputLoggerCallback(TrainerCallback):
    def __init__(self, log_steps):
        super().__init__()
        self.log_steps = log_steps
        self.last_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_steps == 0 and state.global_step > 0:
            current_time = time.time()
            time_delta = current_time - self.last_time
            
            # Calculate per-GPU batch size
            per_gpu_batch_size = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            per_gpu_samples_processed = per_gpu_batch_size * self.log_steps
            per_gpu_samples_per_sec = per_gpu_samples_processed / time_delta
            
            # Calculate aggregate throughput
            total_batch_size = per_gpu_batch_size * args.world_size
            total_samples_processed = total_batch_size * self.log_steps
            total_samples_per_sec = total_samples_processed / time_delta
            
            # Get sequence length from model input
            model = kwargs.get('model')
            if model is not None and hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
                seq_length = model.config.max_position_embeddings
            else:
                # Fallback to max_length from training args if available
                seq_length = getattr(args, 'max_seq_length', None) or 512
            
            # Calculate tokens per second
            per_gpu_tokens_per_sec = per_gpu_samples_per_sec * seq_length
            total_tokens_per_sec = total_samples_per_sec * seq_length
            
            self.last_time = current_time

            rank = args.process_index
            
            # Log per-GPU throughput for all ranks
            if wandb.run is not None:
                wandb.log(
                    {
                        f"throughput/samples_per_sec_rank_{rank}": per_gpu_samples_per_sec,
                        f"throughput/tokens_per_sec_rank_{rank}": per_gpu_tokens_per_sec
                    }, 
                    step=state.global_step
                )
            print(f"[Step {state.global_step}] Rank {rank} Throughput: {per_gpu_samples_per_sec:.2f} samples/sec, {per_gpu_tokens_per_sec:.2f} tokens/sec")
            
            # Log aggregate throughput only from main process
            if rank == 0:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "throughput/samples_per_sec_total": total_samples_per_sec,
                            "throughput/tokens_per_sec_total": total_tokens_per_sec
                        }, 
                        step=state.global_step
                    )
                print(f"[Step {state.global_step}] Total Throughput: {total_samples_per_sec:.2f} samples/sec, {total_tokens_per_sec:.2f} tokens/sec")


def prepare_training_args(config, output_dir):
    strategy = config["distributed"]["strategy"]

    # Common arguments
    args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["training"]["per_device_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "num_train_epochs": config["training"]["epochs"],
        "warmup_steps": config["training"]["warmup_steps"],
        "learning_rate": config["training"]["learning_rate"],
        "bf16": config["training"]["bf16"],
        "fp16": config["training"]["fp16"],
        "remove_unused_columns": False,
        "dataloader_num_workers": config["training"]["num_workers"],
        "report_to": config["training"]["report_to"],
        "gradient_checkpointing": config["training"]["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": config["training"]["save_strategy"],
        "save_steps": config["training"]["save_steps"],
        "logging_steps": config["training"]["logging_steps"],
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": config["training"]["prefetch_factor"]
    }

    if strategy == "fsdp":
        # FSDP-specific arguments
        fsdp_conf = config["distributed"]["fsdp"]
        args_dict["fsdp"] = fsdp_conf["policy"]
        fsdp_inner_config = fsdp_conf["config"].copy()

        if "fsdp_transformer_layer_cls_to_wrap" in fsdp_inner_config:
            args_dict["fsdp_transformer_layer_cls_to_wrap"] = fsdp_inner_config.pop("fsdp_transformer_layer_cls_to_wrap")
            LOGGER.info(f"FSDP Wrapping Layer: {args_dict['fsdp_transformer_layer_cls_to_wrap']}")
        
        if args_dict["gradient_checkpointing"]:
            fsdp_inner_config["activation_checkpointing"] = True
        args_dict["gradient_checkpointing"] = False       
 
        args_dict["fsdp_config"] = fsdp_inner_config
        
        LOGGER.info(f"Training Strategy: FSDP ({args_dict['fsdp']})")
    elif strategy == "ddp":
        # DDP-specific arguments
        ddp_conf = config["distributed"]["ddp"]
        args_dict["ddp_backend"] = ddp_conf["backend"]
        args_dict["ddp_find_unused_parameters"] = ddp_conf["find_unused_parameters"]
        LOGGER.info(f"Training Strategy: DDP (Backend: {args_dict['ddp_backend']})")
    else:
        raise ValueError(f"Unsupported distributed strategy: {strategy}")
    
    training_args = TrainingArguments(**args_dict)
    return training_args

def train(config, accelerator, output_dir):
    # Load tokenizer   
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=torch.bfloat16,
        device_map=None # Let accelerator handle device mapping
    )

    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # Load and process dataset
    subset_path = config["data_mix"]["external_subset_name"]
    num_chunks = config["data_mix"]["num_chunks_to_use"]
    fineweb_data_files = []
    FILES_PER_MAIN_INDEX = config["data_mix"]["files_per_main_index"]
    for i in range(num_chunks):
        main_index = i // FILES_PER_MAIN_INDEX  
        sub_index = i % FILES_PER_MAIN_INDEX   
        main_part = f"{main_index:03d}"  
        sub_part = f"{sub_index:05d}"    
        file_name = f"{subset_path}/{main_part}_{sub_part}.parquet"
        fineweb_data_files.append(file_name)


    fineweb = load_dataset(
        config["data_mix"]["external_dataset_name"],
        #name=config["data_mix"]["external_subset_name"],
        split="train",
        #streaming=True
        data_files=fineweb_data_files
    ).select_columns(["text"])


    dataset = load_dataset(
        "arrow", 
        data_dir=config["data"]["data_folder"], 
        data_files=config["data"]["data_files_pattern"], 
        split="train",
        #streaming=True
    ).select_columns([config["data"]["text_field"]]) # or select_columns("text")
    #dataset = dataset["train"]
    #dataset = dataset.rename_column(config["data"]["text_field"], "text")

    
    # TODO: remove this line for full training
    #dataset = dataset.select(range(10000))

    # Tokenize the text field
    # Silence progress bars on non-main processes
    if not accelerator.is_main_process:
        datasets.utils.logging.disable_progress_bar()

    def tokenize_and_split(examples):
        outputs = tokenizer(
            examples[config["data"]["text_field"]],
            truncation=True,
            max_length=config["training"]["max_length"],
            return_overflowing_tokens=True, # Split long samples
            stride=config["training"]["stride_size"], # Overlap between chunks
            padding=False
        )
        
        if "overflow_to_sample_mapping" in outputs:
            outputs.pop("overflow_to_sample_mapping")
            
        return outputs
    
    #dataset = dataset.map(tokenize_and_split, remove_columns=dataset.column_names)
    
    # Process dataset with main process
    with accelerator.main_process_first():
        fineweb = fineweb.map(
        tokenize_and_split,
        batched=True,
        num_proc=config["training"]["num_workers"],
        batch_size=10000,
        remove_columns=fineweb.column_names, 
        load_from_cache_file=True
        )

        dataset = dataset.map(
            tokenize_and_split,
            batched=True,
            num_proc=config["training"]["preprocessing_num_workers"],
            batch_size=10000,
            remove_columns=dataset.column_names, 
            load_from_cache_file=True
        )

    # Re-enable logging
    if not accelerator.is_main_process:
        datasets.utils.logging.enable_progress_bar()   

    # Mix datasets
    num_fine_web = len(fineweb)
    num_my_data = len(dataset)
    total_samples = num_fine_web + num_my_data
    prob_fine_web = num_fine_web / total_samples
    prob_my_data = num_my_data / total_samples
    # comment this line to use config probabilities
    probabilities = [prob_fine_web, prob_my_data]

    # TODO: remove for full training
    dataset = dataset.select(range(int(1000 * 0.7)))
    fineweb = fineweb.select(range(int(1000 * 0.3)))

    combined_dataset = interleave_datasets(
        [fineweb, dataset],
        probabilities=probabilities,
        seed=config["training"]["seed"], # Usa il seed del training per riproducibilità
        stopping_strategy="first_exhausted" # TODO: if we keep it to put in the config
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=64
    )
    
    # Training arguments
    training_args = prepare_training_args(config, output_dir)
    resource_logging_steps = config["training"]["resource_logging_steps"]    

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            MultiGPUResourcesCallback(resource_logging_steps), 
            ThroughputLoggerCallback(resource_logging_steps)
        ]
    )
    
    # Train the model
    trainer.train()
    return trainer


def init_wandb(config):
    # Login wandb
    wandb.login()

    # Initialize wandb run
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "smiles"),
        name=config["job"]["name"],
        config={
            "model": config["model"]["name"],
            "lr": config["training"]["learning_rate"],
            "batch_size": config["training"]["per_device_batch_size"],
            "epochs": config["training"]["epochs"],
            "data_folder": config["data"]["data_folder"],
            "text_field": config["data"]["text_field"],
            "num_workers": config["training"]["num_workers"],
        }
    )

def train_model(config, output_dir):
    load_dotenv()

    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # Initialize distributed training
    accelerator = Accelerator()

    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        LOGGER.info("Starting fine-tuning.")
        LOGGER.info(f"Distributed: {accelerator.distributed_type}")
        LOGGER.info(f"Process: {accelerator.process_index}/{accelerator.num_processes}")
        LOGGER.info(f"Config: {config}")
        LOGGER.info(f"Output dir: {output_dir}")
        # Initialize wandb
        if config["training"]["report_to"] == "wandb":
            init_wandb(config)

    # Start training
    trainer = train(config, accelerator, output_dir)
    LOGGER.info(f"[RANK {int(os.environ.get('RANK', 0))}] Finished training")
    
    # Save final model
    final_model_dir = f"{output_dir}/final-model" 
    trainer.save_model(final_model_dir)

    if accelerator.is_main_process:
        LOGGER.info(f"Saved model to {final_model_dir}")
