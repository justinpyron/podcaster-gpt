#!/usr/bin/env python3
"""
Script to process MP3 files and generate raw transcripts using OpenAI's transcription API.

This is step 1 in the pipeline:
1. transcribe.py: Convert MP3 files to raw transcript JSONs
2. process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
"""
import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models import RawTranscript, RawTranscriptSegment
from openai import OpenAI


def to_data_url(path: Path) -> str:
    """Convert audio file to base64 data URL."""
    with open(path, "rb") as fh:
        return "data:audio/mp3;base64," + base64.b64encode(fh.read()).decode("utf-8")


def transcribe_audio(
    client: OpenAI,
    audio_path: Path,
    speaker_reference_path: Path,
) -> RawTranscript:
    """
    Transcribe an MP3 audio file using OpenAI API with speaker diarization.

    Args:
        client: OpenAI client instance
        audio_path: Path to the MP3 audio file to transcribe
        speaker_reference_path: Path to the known speaker reference MP3 audio file

    Returns:
        RawTranscript (list of RawTranscriptSegment)
    """
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=audio_file,
            response_format="diarized_json",
            chunking_strategy="auto",
            language="en",
            extra_body={
                "known_speaker_names": ["podcaster"],
                "known_speaker_references": [to_data_url(speaker_reference_path)],
            },
        )

    # Process the transcript segments
    segments = [
        RawTranscriptSegment(
            speaker=segment.speaker,
            text=segment.text,
            start=segment.start,
            end=segment.end,
        )
        for segment in transcript.segments
    ]

    return segments


def process_single_file(
    mp3_file: Path,
    output_dir: Path,
    speaker_reference_path: Path,
    client: OpenAI,
) -> tuple[Path, int | None, str | None]:
    """
    Process a single MP3 file and save its transcript.

    Args:
        mp3_file: Path to the MP3 file to process
        output_dir: Directory where JSON transcript will be saved
        speaker_reference_path: Path to the known speaker reference audio
        client: OpenAI client instance

    Returns:
        Tuple of (mp3_file, segment_count, error_message)
        segment_count is None if error occurred, error_message is None if successful
    """
    try:
        # Transcribe the audio
        segments = transcribe_audio(client, mp3_file, speaker_reference_path)

        # Create output filename (replace .mp3 with .json)
        output_filename = mp3_file.stem + ".json"
        output_path = output_dir / output_filename

        # Save the raw transcript as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [s.model_dump() for s in segments], f, indent=4, ensure_ascii=False
            )

        return (mp3_file, len(segments), None)

    except Exception as e:
        return (mp3_file, None, str(e))


def process_mp3_files(
    input_dir: Path,
    output_dir: Path,
    speaker_reference_path: Path,
    max_workers: int,
) -> None:
    """
    Process all MP3 files in the input directory and save transcripts.

    Args:
        input_dir: Directory containing MP3 files to process
        output_dir: Directory where JSON transcripts will be saved
        speaker_reference_path: Path to the known speaker reference audio
        max_workers: Maximum number of parallel workers
    """
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY_DEFAULT")
    if not api_key:
        raise ValueError("OPENAI_API_KEY_DEFAULT environment variable is not set")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all MP3 files in input directory
    mp3_files = sorted(input_dir.glob("*.mp3"))

    if not mp3_files:
        print(f"No MP3 files found in {input_dir}")
        return

    print(f"Found {len(mp3_files)} MP3 file(s) to process")
    print(f"Using {max_workers} parallel workers\n")

    # Process files in parallel using ThreadPoolExecutor
    completed = 0
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_file,
                mp3_file,
                output_dir,
                speaker_reference_path,
                client,
            ): mp3_file
            for mp3_file in mp3_files
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_file):
            completed += 1
            mp3_file, segment_count, error = future.result()

            if error is None:
                successful += 1
                print(
                    f"[{completed}/{len(mp3_files)}] ✓ {mp3_file.name} "
                    f"({segment_count} segments)"
                )
            else:
                failed += 1
                print(f"[{completed}/{len(mp3_files)}] ❌ {mp3_file.name}: {error}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Total: {len(mp3_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process MP3 files and generate transcripts using OpenAI API"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing MP3 files to process",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JSON transcripts will be saved",
    )

    parser.add_argument(
        "-s",
        "--speaker-reference",
        type=Path,
        required=True,
        help="Path to the known speaker reference audio file (must be 2-10 seconds long)",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=50,
        help="Maximum number of parallel workers (default: 50)",
    )

    args = parser.parse_args()

    # Validate input directory exists
    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    # Validate speaker reference file exists
    if not args.speaker_reference.exists():
        parser.error(f"Speaker reference file does not exist: {args.speaker_reference}")

    # Process the files
    process_mp3_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        speaker_reference_path=args.speaker_reference,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
