"""
Splits a directory of DPO JSON files into training and validation sets.

This script takes a folder of JSON files (each containing a list of DPOExample dicts),
randomly splits the *files* 90/10, and then unpacks all examples into two large
JSON files: train.json and val.json.

Usage:
    python processing/train_val_split.py \
        --input-dir data/examples_dpo/rogan \
        --output-dir data/examples_dpo/rogan-split
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_examples(files: list[Path]) -> list[dict[str, Any]]:
    """Reads a list of JSON files and unpacks their contents into a single list."""
    all_examples = []
    for f in files:
        with open(f, "r") as f_in:
            data = json.load(f_in)
            if isinstance(data, list):
                all_examples.extend(data)
            else:
                all_examples.append(data)
    return all_examples


def split_files(
    files: list[Path], train_ratio: float = 0.9, seed: int = 42
) -> tuple[list[Path], list[Path]]:
    """Shuffles and splits a list of files into two sets."""
    random.seed(seed)
    shuffled = sorted(files)  # Sort first for deterministic shuffling across OSs
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)

    # Ensure at least one file in val if possible
    if split_idx == len(shuffled) and len(shuffled) > 1:
        split_idx = len(shuffled) - 1

    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    parser = argparse.ArgumentParser(
        description="Split DPO JSON files into train and val sets."
    )
    parser.add_argument(
        "-i", "--input-dir", required=True, help="Directory containing input JSON files"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to save train.json and val.json",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_path}")
        return

    print(f"Found {len(json_files)} JSON files.")

    # Split files
    train_files, val_files = split_files(json_files, train_ratio=0.9, seed=args.seed)
    print(f"Split: {len(train_files)} train files, {len(val_files)} val files.")

    # Load and unpack
    train_examples = load_examples(train_files)
    val_examples = load_examples(val_files)
    print(f"Total examples: {len(train_examples)} train, {len(val_examples)} val.")

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    for name, data in [("train.json", train_examples), ("val.json", val_examples)]:
        out_file = output_path / name
        with open(out_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {name} to {out_file}")


if __name__ == "__main__":
    main()
