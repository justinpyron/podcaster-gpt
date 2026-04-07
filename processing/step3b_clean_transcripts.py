#!/usr/bin/env python3
"""
Clean processed transcripts using an LLM.

This is step 3b in the pipeline:
1. step1_chunk_mp3s.py: Split MP3 files into overlapping chunks
2. step2_transcribe.py: Convert MP3 files to raw transcript JSONs
3. step3_process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
3b. step3b_clean_transcripts.py: Clean processed transcripts using an LLM
4. step4_create_sft_examples.py: Convert processed transcripts to SFT examples
5. step5_create_dpo_examples.py: Generate rejected completions for DPO training data
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from data_types import Message, ProcessedTranscript
from jinja2 import Template
from openai import OpenAI

TEMPLATE = "Clean this transcript: {{ raw_transcript }}"

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def clean_transcript(
    messages: ProcessedTranscript,
    client: OpenAI,
    model: str = "gpt-5.4-nano",
    reasoning_effort: ReasoningEffort = "medium",
) -> ProcessedTranscript:
    """
    Clean a processed transcript by passing it through an LLM.

    Args:
        messages: ProcessedTranscript (list of Message).
        client: OpenAI client instance.
        model: Model to use.
        reasoning_effort: Reasoning effort level.

    Returns:
        ProcessedTranscript with cleaned messages.
    """
    prompt = Template(TEMPLATE).render(
        raw_transcript=json.dumps([m.model_dump() for m in messages])
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort},
    )
    cleaned = json.loads(response.output_text)
    return [Message.model_validate(m) for m in cleaned]


def process_single_file(
    json_file: Path,
    output_dir: Path,
    client: OpenAI,
    model: str,
    reasoning_effort: ReasoningEffort,
) -> tuple[Path, int | None, str | None]:
    """
    Process a single transcript file.

    Args:
        json_file: Path to the input JSON file.
        output_dir: Directory where cleaned transcript will be saved.
        client: OpenAI client instance.
        model: Model to use.
        reasoning_effort: Reasoning effort level.

    Returns:
        Tuple of (json_file, message_count, error_message).
        message_count is None if error occurred, error_message is None if successful.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        messages = [Message.model_validate(m) for m in raw_data]
        cleaned = clean_transcript(messages, client, model, reasoning_effort)

        output_path = output_dir / json_file.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [m.model_dump() for m in cleaned],
                f,
                indent=4,
                ensure_ascii=False,
            )

        return (json_file, len(cleaned), None)

    except Exception as e:
        return (json_file, None, str(e))


def process_all_files(
    input_dir: Path,
    output_dir: Path,
    model: str,
    reasoning_effort: ReasoningEffort,
    max_workers: int,
) -> None:
    """
    Clean all processed transcript JSON files in the input directory.

    Args:
        input_dir: Directory containing processed transcript JSON files.
        output_dir: Directory where cleaned transcript JSON files will be saved.
        model: OpenAI model to use.
        reasoning_effort: Reasoning effort level.
        max_workers: Maximum number of parallel workers.
    """
    api_key = os.getenv("OPENAI_API_KEY_DEFAULT")
    if not api_key:
        raise ValueError("OPENAI_API_KEY_DEFAULT environment variable is not set")

    client = OpenAI(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process")
    print(f"Model: {model}")
    print(f"Reasoning effort: {reasoning_effort}")
    print(f"Workers: {max_workers}\n")

    completed = 0
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                process_single_file,
                json_file,
                output_dir,
                client,
                model,
                reasoning_effort,
            ): json_file
            for json_file in json_files
        }

        for future in as_completed(future_to_file):
            completed += 1
            json_file, message_count, error = future.result()

            if error is None:
                successful += 1
                print(
                    f"[{completed}/{len(json_files)}] ✓ {json_file.name} "
                    f"({message_count} messages)"
                )
            else:
                failed += 1
                print(f"[{completed}/{len(json_files)}] ❌ {json_file.name}: {error}")

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"  Total: {len(json_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean processed transcripts using an LLM"
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
        help="Directory where cleaned transcript JSON files will be saved",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-5.4-nano",
        help='OpenAI model to use (default: "gpt-5.4-nano")',
    )

    parser.add_argument(
        "-r",
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=REASONING_EFFORTS,
        help='Reasoning effort level (default: "medium")',
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=50,
        help="Maximum number of parallel workers (default: 50)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
