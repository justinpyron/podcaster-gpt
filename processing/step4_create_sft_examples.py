#!/usr/bin/env python3
"""
Create supervised fine-tuning (SFT) examples from processed transcripts.

This is step 4 in the pipeline:
1. step1_chunk_mp3s.py: Split MP3 files into overlapping chunks
2. step2_transcribe.py: Convert MP3 files to raw transcript JSONs
3. step3_process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
4. step4_create_sft_examples.py: Convert processed transcripts to SFT examples
5. step5_create_dpo_examples.py: Generate rejected completions for DPO training data
"""
import argparse
import json
from pathlib import Path

from data_types import Message, SFTExample


def filter_messages(
    example: SFTExample,
    min_completion_words: int,
    min_avg_prompt_words: float,
) -> SFTExample | None:
    """
    Filter an SFT example based on criteria like word count.

    Args:
        example: The SFTExample to check.
        min_completion_words: Minimum word count for the completion.
        min_avg_prompt_words: Minimum average word count for prompt messages.

    Returns:
        The example if it passes filtering, otherwise None.
    """
    # 1. Check completion word count
    completion_content = example.completion[0].content
    if len(completion_content.split()) < min_completion_words:
        return None

    # 2. Check average prompt word count
    prompt_word_counts = [len(m.content.split()) for m in example.prompt]
    avg_prompt_words = sum(prompt_word_counts) / len(prompt_word_counts)

    if avg_prompt_words < min_avg_prompt_words:
        return None

    return example


def create_finetune_examples(
    messages: list[Message],
    min_completion_words: int = 5,
    min_avg_prompt_words: float = 0.0,
) -> list[SFTExample]:
    """
    Transform a conversation into training examples for fine-tuning an LLM.

    For each assistant message, creates an SFTExample where prompt is all
    preceding messages and completion is the assistant message.
    Ensures that the prompt for every example starts with a user message.

    Args:
        messages: List of Message objects.
        min_completion_words: Minimum words for completion.
        min_avg_prompt_words: Minimum average words for prompt messages.

    Returns:
        List of SFTExample objects.
    """
    # Ensure conversation starts with a user message
    start_idx = next((i for i, m in enumerate(messages) if m.role == "user"), None)
    if start_idx is None:
        return []
    messages = messages[start_idx:]

    # Create the initial set of SFTExample objects
    examples = [
        SFTExample(prompt=messages[:i], completion=[m])
        for i, m in enumerate(messages)
        if m.role == "assistant" and i > 0
    ]

    # Filter the examples
    filtered_examples = [
        ex
        for ex in examples
        if filter_messages(ex, min_completion_words, min_avg_prompt_words) is not None
    ]

    return filtered_examples


def process_all_files(
    input_dir: Path,
    output_dir: Path,
    min_completion_words: int = 5,
    min_avg_prompt_words: float = 0.0,
) -> None:
    """
    Process all JSON transcript files and save SFT examples.

    Args:
        input_dir: Directory containing processed transcript JSON files.
        output_dir: Directory where SFT example JSON files will be saved.
        min_completion_words: Minimum words for completion.
        min_avg_prompt_words: Minimum average words for prompt messages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process")
    print(f"Filtering completion < {min_completion_words} words")
    print(f"Filtering avg prompt < {min_avg_prompt_words} words\n")

    successful = 0
    failed = 0
    total_examples = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            messages = [Message.model_validate(m) for m in raw_data]
            examples = create_finetune_examples(
                messages,
                min_completion_words=min_completion_words,
                min_avg_prompt_words=min_avg_prompt_words,
            )

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

    parser.add_argument(
        "-w",
        "--min-completion-words",
        type=int,
        default=5,
        help="Minimum word count for the completion (default: 5)",
    )

    parser.add_argument(
        "-p",
        "--min-avg-prompt-words",
        type=float,
        default=0.0,
        help="Minimum average word count for prompt messages (default: 0.0)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_completion_words=args.min_completion_words,
        min_avg_prompt_words=args.min_avg_prompt_words,
    )


if __name__ == "__main__":
    main()
