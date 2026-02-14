#!/usr/bin/env python3
"""
Create supervised fine-tuning (SFT) examples from processed transcripts.

This is step 3 in the pipeline:
1. transcribe.py: Convert MP3 files to raw transcript JSONs
2. process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
3. create_sft_examples.py: Convert processed transcripts to (prompt, completion) examples
"""
import argparse
import json
from pathlib import Path

from transcript_types import ProcessedTranscriptMessage, SftExample


def create_finetune_examples(
    messages: list[ProcessedTranscriptMessage],
) -> list[SftExample]:
    """
    Transform a conversation into training examples for fine-tuning an LLM.

    For each assistant message, creates an SftExample where prompt is all
    preceding messages and completion is the assistant message.

    Args:
        messages: List of ProcessedTranscriptMessage objects.

    Returns:
        List of SftExample objects.
    """
    return [
        SftExample(prompt=messages[:i], completion=[m])
        for i, m in enumerate(messages)
        if m.role == "assistant" and i > 0
    ]


def process_all_files(input_dir: Path, output_dir: Path) -> None:
    """
    Process all JSON transcript files and save SFT examples.

    Args:
        input_dir: Directory containing processed transcript JSON files.
        output_dir: Directory where SFT example JSON files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process\n")

    successful = 0
    failed = 0
    total_examples = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            messages = [ProcessedTranscriptMessage.model_validate(m) for m in raw_data]
            examples = create_finetune_examples(messages)

            output_path = output_dir / json_file.name
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [e.model_dump() for e in examples],
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            successful += 1
            total_examples += len(examples)
            print(f"✓ {json_file.name} ({len(examples)} examples)")

        except Exception as e:
            failed += 1
            print(f"✗ {json_file.name}: {e}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Files: {successful} successful, {failed} failed")
    print(f"  Total SFT examples: {total_examples}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create SFT training examples from processed transcripts"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing processed transcript JSON files",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where SFT example JSON files will be saved",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
