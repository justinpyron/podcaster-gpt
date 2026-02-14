"""
DPO training with LoRA on Modal.

Loads a base model plus an existing SFT LoRA adapter, then trains the adapter
further with Direct Preference Optimization. The SFT adapter is loaded twice:
once as the trainable policy ("train") and once as the frozen reference
("reference"), following the TRL "Option 3" pattern for PEFT-based DPO.

Logs are sent to Weights & Biases.

Usage:
    modal run train_dpo.py \
        --model-path <path-in-volume> \
        --adapter-path <path-in-volume> \
        --data-path-train <path-in-volume> \
        --data-path-val <path-in-volume> \
        --name <run-name> \
        [--learning-rate 5e-5] \
        [--num-epochs 1] \
        [--batch-size 4] \
        [--max-length 2048] \
        [--max-prompt-length 1800] \
        [--beta 0.1] \
        [--gradient-accumulation-steps 4] \
        [--logging-steps 0.1] \
        [--eval-steps 0.5]
"""

import json
from pathlib import Path

import modal

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "impersonate-gpt-dpo"
VOLUME_NAME = "impersonate-gpt"
VOLUME_MOUNT_PATH = "/data"
WEIGHTS_DIR = "weights_dpo"
GPU = "A10"
TIMEOUT_SECONDS = 24 * 60 * 60
WANDB_ENTITY = "pyron"
WANDB_PROJECT = "impersonate-gpt-dpo"

# =============================================================================
# Modal Setup
# =============================================================================

app = modal.App(APP_NAME)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "wandb",
)

volume = modal.Volume.from_name(VOLUME_NAME)


# =============================================================================
# Training
# =============================================================================


def load_dpo_dataset(path: str):
    """Load DPO examples from a path (file or directory) into a HuggingFace Dataset.

    Each JSON file contains a list of DPOExamples with keys:
    "prompt", "chosen", "rejected" — each being a list of message dicts.
    """
    from datasets import Dataset

    all_examples = []
    p = Path(path)
    if p.is_file():
        files = [p]
    else:
        files = sorted(list(p.glob("*.json")))

    for filepath in files:
        with open(filepath, "r") as f:
            all_examples.extend(json.load(f))

    return Dataset.from_dict(
        {
            "prompt": [ex["prompt"] for ex in all_examples],
            "chosen": [ex["chosen"] for ex in all_examples],
            "rejected": [ex["rejected"] for ex in all_examples],
        }
    )


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    model_path: str,
    adapter_path: str,
    data_path_train: str,
    data_path_val: str,
    name: str,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    max_prompt_length: int,
    beta: float,
    logging_steps: float,
    eval_steps: float,
):
    """Run DPO training with LoRA Option 3 on a frozen base model + SFT adapter."""
    import os
    from datetime import datetime, timezone

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    # Generate run name with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{name}_{timestamp}"
    output_dir_full = f"{VOLUME_MOUNT_PATH}/{WEIGHTS_DIR}/{run_name}"

    # Configure wandb
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # Data
    dataset_train = load_dpo_dataset(f"{VOLUME_MOUNT_PATH}/{data_path_train}")
    dataset_val = load_dpo_dataset(f"{VOLUME_MOUNT_PATH}/{data_path_val}")

    # Tokenizer
    model_path_full = f"{VOLUME_MOUNT_PATH}/{model_path}"
    tokenizer = AutoTokenizer.from_pretrained(model_path_full)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model — load base then attach the SFT adapter twice (Option 3)
    model = AutoModelForCausalLM.from_pretrained(model_path_full)
    adapter_path_full = f"{VOLUME_MOUNT_PATH}/{adapter_path}"
    model = PeftModel.from_pretrained(
        model,
        adapter_path_full,
        is_trainable=True,
        adapter_name="train",
    )
    model.load_adapter(adapter_path_full, adapter_name="reference")

    # Training config
    config_training = DPOConfig(
        output_dir=output_dir_full,
        run_name=run_name,
        # LoRA Option 3
        model_adapter_name="train",
        ref_adapter_name="reference",
        # Hyperparameters
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        # Logging & evaluation
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        # Checkpointing
        save_strategy="best",
        load_best_model_at_end=True,
        save_total_limit=3,
        # Reporting
        report_to="wandb",
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        args=config_training,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir_full)
    volume.commit()


# =============================================================================
# CLI Entrypoint
# =============================================================================


@app.local_entrypoint()
def main(
    model_path: str,
    adapter_path: str,
    data_path_train: str,
    data_path_val: str,
    name: str,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    max_prompt_length: int = 1800,
    beta: float = 0.1,
    logging_steps: float = 0.1,
    eval_steps: float = 0.1,
):
    """Launch DPO training on Modal. All paths are relative to the volume root."""
    print("=" * 80)
    print("Launching DPO training job on Modal...")
    print(f"  Base model: {model_path}")
    print(f"  SFT adapter: {adapter_path}")
    print(f"  Train data: {data_path_train}")
    print(f"  Val data: {data_path_val}")
    print(f"  Run name: {name}")
    print(f"  DPO beta: {beta}")
    print(
        f"  Training: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}"
    )
    print("-" * 40)

    train.remote(
        model_path=model_path,
        adapter_path=adapter_path,
        data_path_train=data_path_train,
        data_path_val=data_path_val,
        name=name,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        beta=beta,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
    )
