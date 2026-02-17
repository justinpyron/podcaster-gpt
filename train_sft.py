"""
SFT training with LoRA on Modal.

Trains LoRA adapters on a frozen base model. Logs are sent to Weights & Biases.

Usage:
    modal run train_sft.py \
        --model-path <path-in-volume> \
        --data-path-train <path-in-volume> \
        --data-path-val <path-in-volume> \
        --name <run-name> \
        [--lora-r 16] \
        [--lora-alpha 32] \
        [--learning-rate 0.0002] \
        [--num-epochs 1] \
        [--batch-size 4] \
        [--max-length 2048] \
        [--gradient-accumulation-steps 4] \
        [--logging-steps 0.1] \
        [--eval-steps 0.1]
"""

import json
from pathlib import Path

import modal

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "podcaster-gpt-sft"
VOLUME_NAME = "podcaster-gpt"
VOLUME_MOUNT_PATH = "/data"
WEIGHTS_DIR = "weights_sft"
GPU = "A100-80GB"
TIMEOUT_SECONDS = 24 * 60 * 60
WANDB_ENTITY = "pyron"
WANDB_PROJECT = "podcaster-gpt-sft"

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


def load_sft_dataset(path: str):
    """Load a JSON file or directory of SFTExamples into a HuggingFace Dataset."""
    from datasets import Dataset

    examples = []
    p = Path(path)
    if p.is_file():
        files = [p]
    else:
        files = sorted(list(p.glob("*.json")))

    for filepath in files:
        with open(filepath, "r") as f:
            examples.extend(json.load(f))

    return Dataset.from_dict(
        {
            "prompt": [ex["prompt"] for ex in examples],
            "completion": [ex["completion"] for ex in examples],
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
    data_path_train: str,
    data_path_val: str,
    name: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    logging_steps: float,
    eval_steps: float,
):
    """Run SFT training with LoRA on a frozen base model."""
    import os
    from datetime import datetime, timezone

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Generate run name with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{name}_{timestamp}"
    output_dir_full = f"{VOLUME_MOUNT_PATH}/{WEIGHTS_DIR}/{run_name}"

    # Configure wandb
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # Data
    dataset_train = load_sft_dataset(f"{VOLUME_MOUNT_PATH}/{data_path_train}")
    dataset_val = load_sft_dataset(f"{VOLUME_MOUNT_PATH}/{data_path_val}")

    # Model and tokenizer
    model_path_full = f"{VOLUME_MOUNT_PATH}/{model_path}"
    tokenizer = AutoTokenizer.from_pretrained(model_path_full)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path_full)

    # LoRA
    config_lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    config_training = SFTConfig(
        output_dir=output_dir_full,
        run_name=run_name,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        completion_only_loss=True,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="best",
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to="wandb",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=config_training,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        processing_class=tokenizer,
        peft_config=config_lora,
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
    data_path_train: str,
    data_path_val: str,
    name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    logging_steps: float = 0.1,
    eval_steps: float = 0.1,
):
    """Launch SFT training on Modal. All paths are relative to the volume root."""
    print("=" * 80)
    print("Launching SFT training job on Modal...")
    print(f"  Model: {model_path}")
    print(f"  Train data: {data_path_train}")
    print(f"  Val data: {data_path_val}")
    print(f"  Run name: {name}")
    print(f"  LoRA config: r={lora_r}, alpha={lora_alpha}")
    print(
        f"  Training: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}"
    )
    print("-" * 40)

    train.remote(
        model_path=model_path,
        data_path_train=data_path_train,
        data_path_val=data_path_val,
        name=name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
    )
