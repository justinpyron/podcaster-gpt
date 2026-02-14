#!/usr/bin/env python3
"""
Process raw transcripts and generate formatted messages for training.

This is step 3 in the pipeline:
1. chunk_mp3s.py: Split MP3 files into overlapping chunks
2. transcribe.py: Convert MP3 files to raw transcript JSONs
3. process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
4. create_sft_examples.py: Convert processed transcripts to SFT examples
5. create_dpo_examples.py: Generate rejected completions for DPO training data
"""
import argparse
import json
from pathlib import Path

from data_types import Message, ProcessedTranscript, RawTranscript, TranscriptSegment


def drop_first_and_last(segments: RawTranscript) -> RawTranscript:
    """
    Return a new list with the first and last segments removed. This is to discard
    clips that may be incomplete because they crossed a chunk boundary.

    Args:
        segments: RawTranscript (list of TranscriptSegment).

    Returns:
        RawTranscript with first and last segments dropped. If the list has 2 or fewer segments, returns an empty list.
    """
    return segments[1:-1]


def merge_adjacent_speakers(
    segments: RawTranscript,
    threshold_seconds: float = 1,
    separator: str = "\n",
) -> RawTranscript:
    """
    Merge adjacent segments that have the same speaker.
    If the time between segments is >= threshold_seconds, separate by the given separator; otherwise, concatenate directly.

    Args:
        segments: RawTranscript (list of TranscriptSegment).
        threshold_seconds: float, threshold in seconds to insert the separator between merged segments
        separator: str, string inserted between merged segments when the time gap is >= threshold

    Returns:
        RawTranscript with adjacent same-speaker segments merged.
    """
    if not segments:
        return []

    merged = []
    current = segments[0].model_copy()

    for i in range(1, len(segments)):
        next_seg = segments[i]
        if next_seg.speaker == current.speaker:
            time_gap = next_seg.start - current.end
            if time_gap < threshold_seconds:
                current.text += next_seg.text
            else:
                current.text += separator + next_seg.text
            current.end = next_seg.end
        else:
            merged.append(current)
            current = next_seg.model_copy()

    merged.append(current)

    return merged


def clean_messages(segments: RawTranscript) -> RawTranscript:
    """
    Clean segments by removing leading and trailing whitespace from the 'text' field.

    Args:
        segments: RawTranscript (list of TranscriptSegment).

    Returns:
        RawTranscript with text fields stripped of leading and trailing whitespace.
    """
    return [s.model_copy(update={"text": s.text.strip()}) for s in segments]


def relabel_non_podcasters_as_guest(segments: RawTranscript) -> RawTranscript:
    """
    Relabel the 'speaker' value of all segments to 'guest' if it is not equal to 'podcaster'.

    Args:
        segments: RawTranscript (list of TranscriptSegment).

    Returns:
        RawTranscript with the 'speaker' field set to 'guest' if it was not 'podcaster'.
    """
    return [
        s.model_copy(
            update={"speaker": "podcaster" if s.speaker == "podcaster" else "guest"}
        )
        for s in segments
    ]


def transform_into_chatbot_format(segments: RawTranscript) -> ProcessedTranscript:
    """
    Transform raw transcript segments into chatbot format.

    Args:
        segments: RawTranscript (list of TranscriptSegment).

    Returns:
        ProcessedTranscript (list of Message).
    """
    return [
        Message(
            role="assistant" if s.speaker == "podcaster" else "user",
            content=s.text,
        )
        for s in segments
    ]


def process_transcript(
    segments: RawTranscript,
    merge_threshold_seconds: float = 1.0,
    merge_separator: str = "\n\n",
) -> ProcessedTranscript:
    """
    Process a raw transcript and generate formatted messages for training.

    Args:
        segments: RawTranscript (list of TranscriptSegment).
        merge_threshold_seconds: Time gap (in seconds) between same-speaker segments
            above which the merge separator is inserted.
        merge_separator: String inserted between merged segments when the time gap
            exceeds the threshold.

    Returns:
        ProcessedTranscript (list of Message).
    """
    segments = drop_first_and_last(segments)
    segments = relabel_non_podcasters_as_guest(segments)
    segments = merge_adjacent_speakers(
        segments,
        threshold_seconds=merge_threshold_seconds,
        separator=merge_separator,
    )
    segments = clean_messages(segments)
    return transform_into_chatbot_format(segments)


def process_all_files(
    input_dir: Path,
    output_dir: Path,
    merge_threshold_seconds: float = 1.0,
    merge_separator: str = "\n\n",
) -> None:
    """
    Process all JSON transcript files in the input directory and save processed versions.

    Args:
        input_dir: Directory containing raw transcript JSON files
        output_dir: Directory where processed transcript JSON files will be saved
        merge_threshold_seconds: Time gap threshold for inserting separator between merged segments
        merge_separator: String inserted between merged segments when time gap exceeds threshold
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process\n")

    successful = 0
    failed = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            segments = [TranscriptSegment.model_validate(item) for item in raw_data]
            processed_messages = process_transcript(
                segments,
                merge_threshold_seconds=merge_threshold_seconds,
                merge_separator=merge_separator,
            )

            output_path = output_dir / json_file.name
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [m.model_dump() for m in processed_messages],
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            successful += 1
            print(f"✓ {json_file.name} ({len(processed_messages)} messages)")

        except Exception as e:
            failed += 1
            print(f"❌ {json_file.name}: {e}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Total: {len(json_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process raw transcript JSONs into formatted training data"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw transcript JSON files",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where processed transcript JSON files will be saved",
    )

    parser.add_argument(
        "-mt",
        "--merge-threshold-seconds",
        type=float,
        default=1.0,
        help="Time gap in seconds above which a separator is inserted when merging "
        "adjacent same-speaker segments (default: 1.0)",
    )

    parser.add_argument(
        "-ms",
        "--merge-separator",
        type=str,
        default="\n\n",
        help="String inserted between merged segments when the time gap exceeds the "
        'threshold (default: "\\n\\n")',
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        merge_threshold_seconds=args.merge_threshold_seconds,
        merge_separator=args.merge_separator,
    )


if __name__ == "__main__":
    main()
