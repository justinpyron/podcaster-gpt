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


def main():
    parser = argparse.ArgumentParser(
        description="Split DPO JSON files into train and val sets."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input JSON files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save train.json and val.json",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    # Find all JSON files
    json_files = sorted(list(input_path.glob("*.json")))
    if not json_files:
        print(f"No JSON files found in {input_path}")
        return

    print(f"Found {len(json_files)} JSON files.")

    # Shuffle files
    random.shuffle(json_files)

    # 90/10 split
    split_idx = int(len(json_files) * 0.9)
    # Ensure at least one file in val if there are enough files
    if split_idx == len(json_files) and len(json_files) > 1:
        split_idx = len(json_files) - 1

    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    print(
        f"Splitting into {len(train_files)} train files and {len(val_files)} val files."
    )

    def collect_examples(files):
        examples = []
        for f in files:
            with open(f, "r") as f_in:
                data = json.load(f_in)
                if isinstance(data, list):
                    examples.extend(data)
                else:
                    examples.append(data)
        return examples

    train_examples = collect_examples(train_files)
    val_examples = collect_examples(val_files)

    print(f"Total examples: {len(train_examples)} train, {len(val_examples)} val.")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "train.json", "w") as f:
        json.dump(train_examples, f, indent=4)

    with open(output_path / "val.json", "w") as f:
        json.dump(val_examples, f, indent=4)

    print(f"Saved split to {output_path}")


if __name__ == "__main__":
    main()
