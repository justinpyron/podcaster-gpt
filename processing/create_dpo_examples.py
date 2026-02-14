#!/usr/bin/env python3
"""
Create Direct Preference Optimization (DPO) examples from SFT examples.

This is step 4 in the pipeline:
1. transcribe.py: Convert MP3 files to raw transcript JSONs
2. process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
3. create_sft_examples.py: Convert processed transcripts to SFT examples
4. create_dpo_examples.py: Generate rejected completions for DPO training data
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from together import Together
from transcript_types import DPOExample, Message, SFTExample

SYSTEM_PROMPT = """You ("assistant") are having a verbal conversation with a friend ("user"). Continue the conversation.

Return only the exact text you say in the conversation. So, do not return any other text: no reasoning, no thinking, etc.
"""


def generate_rejected_completion(
    client: Together,
    prompt: list[Message],
    model: str,
) -> Message:
    """
    Generate a fake (rejected) completion for a given prompt using Together AI.

    Args:
        client: Together AI client instance.
        prompt: List of preceding messages.
        model: Together AI model name to use.

    Returns:
        A Message object with role "assistant" containing the generated text.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        m.model_dump() for m in prompt
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    content = response.choices[0].message.content.strip()
    return Message(role="assistant", content=content)


def process_single_example(
    client: Together,
    sft_example: SFTExample,
    model: str,
    min_words: int,
) -> DPOExample | None:
    """
    Process a single SFT example into a DPO example if it meets criteria.

    Args:
        client: Together AI client.
        sft_example: The SFT example to process.
        model: Model to use for generation.
        min_words: Minimum word count for the chosen completion.

    Returns:
        DPOExample if processed, None if filtered out.
    """
    # Filter by word count of the chosen completion
    chosen_content = sft_example.completion[0].content
    word_count = len(chosen_content.split())

    if word_count < min_words:
        return None

    # Generate rejected completion
    rejected_message = generate_rejected_completion(client, sft_example.prompt, model)

    return DPOExample(
        prompt=sft_example.prompt,
        chosen=sft_example.completion,
        rejected=[rejected_message],
    )


def process_all_files(
    input_dir: Path,
    output_dir: Path,
    model: str,
    min_words: int,
    max_workers: int,
) -> None:
    """
    Process all SFT example files and generate DPO examples.

    Args:
        input_dir: Directory containing SFT example JSON files.
        output_dir: Directory where DPO example JSON files will be saved.
        model: Together AI model to use.
        min_words: Minimum words for chosen completion.
        max_workers: Maximum parallel workers.
    """
    # Get API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is not set")

    client = Together(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process")
    print(f"Using model: {model}")
    print(f"Filtering examples with < {min_words} words")
    print(f"Using {max_workers} workers\n")

    successful_files = 0
    failed_files = 0
    total_dpo_examples = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            sft_examples = [SFTExample.model_validate(ex) for ex in raw_data]
            dpo_examples = []

            # Process examples in parallel for each file
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ex = {
                    executor.submit(
                        process_single_example, client, ex, model, min_words
                    ): ex
                    for ex in sft_examples
                }

                for future in as_completed(future_to_ex):
                    result = future.result()
                    if result:
                        dpo_examples.append(result)

            if dpo_examples:
                output_path = output_dir / json_file.name
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [ex.model_dump() for ex in dpo_examples],
                        f,
                        indent=4,
                        ensure_ascii=False,
                    )
                total_dpo_examples += len(dpo_examples)
                print(f"✓ {json_file.name} ({len(dpo_examples)} DPO examples)")
            else:
                print(f"! {json_file.name} (0 examples after filtering)")

            successful_files += 1

        except Exception as e:
            failed_files += 1
            print(f"✗ {json_file.name}: {e}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Files: {successful_files} successful, {failed_files} failed")
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
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Together AI model name to use (default: openai/gpt-oss-120b)",
    )

    parser.add_argument(
        "-w",
        "--min-words",
        type=int,
        default=5,
        help="Minimum word count for the chosen completion (default: 5)",
    )

    parser.add_argument(
        "-t",
        "--workers",
        type=int,
        default=10,
        help="Maximum parallel workers (default: 10)",
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
        min_words=args.min_words,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
