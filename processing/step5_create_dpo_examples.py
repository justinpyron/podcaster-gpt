#!/usr/bin/env python3
"""
Create Direct Preference Optimization (DPO) examples from SFT examples.

This is step 5 in the pipeline:
1. step1_chunk_mp3s.py: Split MP3 files into overlapping chunks
2. step2_transcribe.py: Convert MP3 files to raw transcript JSONs
3. step3_process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
4. step4_create_sft_examples.py: Convert processed transcripts to SFT examples
5. step5_create_dpo_examples.py: Generate rejected completions for DPO training data
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data_types import DPOExample, Message, SFTExample
from together import Together

SYSTEM_PROMPT = """You ("assistant") are having a verbal conversation with a friend ("user"). Continue the conversation.

Return only the exact text you say in the conversation. So, do not return any other text: no reasoning, no thinking, etc.
"""


def generate_rejected_completion(
    client: Together,
    prompt: list[Message],
    model_id: str,
) -> Message:
    """
    Generate a fake (rejected) completion for a given prompt using Together AI.

    Args:
        client: Together AI client instance.
        prompt: List of preceding messages.
        model_id: Together AI model name to use.

    Returns:
        A Message object with role "assistant" containing the generated text.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        m.model_dump() for m in prompt
    ]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )

    content = response.choices[0].message.content.strip()
    return Message(role="assistant", content=content)


def process_single_example(
    client: Together,
    sft_example: SFTExample,
    model_id: str,
    output_filename: str,
) -> tuple[DPOExample | None, str]:
    """
    Process a single SFT example into a DPO example.

    Args:
        client: Together AI client.
        sft_example: The SFT example to process.
        model_id: Model ID to use for generation.
        output_filename: Name of the file this example belongs to.

    Returns:
        Tuple of (DPOExample or None, output_filename).
    """
    # Generate rejected completion
    rejected_message = generate_rejected_completion(
        client, sft_example.prompt, model_id
    )

    dpo_ex = DPOExample(
        prompt=sft_example.prompt,
        chosen=sft_example.completion,
        rejected=[rejected_message],
    )
    return dpo_ex, output_filename


def process_all_files(
    input_dir: Path,
    output_dir: Path,
    model_id: str,
    max_workers: int,
) -> None:
    """
    Process all SFT example files and generate DPO examples.

    Args:
        input_dir: Directory containing SFT example JSON files.
        output_dir: Directory where DPO example JSON files will be saved.
        model_id: Together AI model ID to use.
        max_workers: Maximum parallel workers.
    """
    client = Together()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process")

    # Step 1: Collect all work items from all files
    all_work_items = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            sft_examples = [SFTExample.model_validate(ex) for ex in raw_data]
            for ex in sft_examples:
                all_work_items.append((ex, json_file.name))
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")

    if not all_work_items:
        print("No examples found to process.")
        return

    print(f"Collected {len(all_work_items)} total examples to process")
    print(f"Using model ID: {model_id}")
    print(f"Using {max_workers} workers\n")

    # Step 2: Process all examples in parallel across a single thread pool
    results_by_file = defaultdict(list)
    total_processed = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {
            executor.submit(process_single_example, client, ex, model_id, fname): (
                ex,
                fname,
            )
            for ex, fname in all_work_items
        }

        for future in as_completed(future_to_info):
            _, original_fname = future_to_info[future]
            try:
                dpo_ex, res_fname = future.result()
                if dpo_ex:
                    results_by_file[res_fname].append(dpo_ex)
            except Exception as e:
                total_failed += 1
                print(f"❌ Error processing example from {original_fname}: {e}")

            total_processed += 1
            if total_processed % 10 == 0 or total_processed == len(all_work_items):
                print(
                    f"\rProgress: {total_processed}/{len(all_work_items)} examples processed "
                    f"({total_failed} failed)...",
                    end="",
                    flush=True,
                )

    print("\n\nWriting output files...")

    # Step 3: Write results back to their respective files
    successful_files = 0
    total_dpo_examples = 0

    for fname in sorted(results_by_file.keys()):
        examples = results_by_file[fname]
        if not examples:
            continue

        try:
            output_path = output_dir / fname
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [ex.model_dump() for ex in examples],
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            total_dpo_examples += len(examples)
            successful_files += 1
        except Exception as e:
            print(f"❌ Error writing {fname}: {e}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Examples: {total_processed} total, {total_failed} failed")
    print(f"  Files: {successful_files} output files created")
    print(f"  Total DPO examples: {total_dpo_examples}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create DPO training examples by generating rejected completions"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing SFT example JSON files",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where DPO example JSON files will be saved",
    )

    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
        default="openai/gpt-oss-120b",
        help="Together AI model ID to use (default: openai/gpt-oss-120b)",
    )

    parser.add_argument(
        "-t",
        "--workers",
        type=int,
        default=50,
        help="Maximum parallel workers (default: 50)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
