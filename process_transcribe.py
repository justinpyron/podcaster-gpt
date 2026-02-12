#!/usr/bin/env python3
"""
Script to process MP3 files and generate transcripts using OpenAI's transcription API.
"""
import argparse
import base64
import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


def to_data_url(path: Path) -> str:
    """Convert audio file to base64 data URL."""
    with open(path, "rb") as fh:
        return "data:audio/mp3;base64," + base64.b64encode(fh.read()).decode("utf-8")


def transcribe_audio(
    client: OpenAI,
    audio_path: Path,
    speaker_reference_path: Path,
) -> List[Dict[str, str]]:
    """
    Transcribe an MP3 audio file using OpenAI API with speaker diarization.

    Args:
        client: OpenAI client instance
        audio_path: Path to the MP3 audio file to transcribe
        speaker_reference_path: Path to the known speaker reference MP3 audio file

    Returns:
        List of message dictionaries with 'speaker' and 'text' keys
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
    messages = [
        {
            "speaker": segment.speaker,
            "text": segment.text,
        }
        for segment in transcript.segments
    ]

    return messages


def process_mp3_files(
    input_dir: Path,
    output_dir: Path,
    speaker_reference_path: Path,
    api_key: str,
) -> None:
    """
    Process all MP3 files in the input directory and save transcripts.

    Args:
        input_dir: Directory containing MP3 files to process
        output_dir: Directory where JSON transcripts will be saved
        speaker_reference_path: Path to the known speaker reference audio
        api_key: OpenAI API key
    """
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

    # Process each MP3 file
    for mp3_file in mp3_files:
        print(f"\nProcessing: {mp3_file.name}")

        try:
            # Transcribe the audio
            messages = transcribe_audio(client, mp3_file, speaker_reference_path)

            # Create output filename (replace .mp3 with .json)
            output_filename = mp3_file.stem + ".json"
            output_path = output_dir / output_filename

            # Save the processed transcript as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=4, ensure_ascii=False)

            print(f"  ✓ Saved transcript to: {output_path}")
            print(f"  ✓ Found {len(messages)} segments")

        except Exception as e:
            print(f"  ❌ Error processing {mp3_file.name}: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process MP3 files and generate transcripts using OpenAI API"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing MP3 files to process",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JSON transcripts will be saved",
    )

    parser.add_argument(
        "--speaker-reference",
        type=Path,
        required=True,
        help="Path to the known speaker reference audio file (must be 2-10 seconds long)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY_DEFAULT environment variable)",
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

    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv("OPENAI_API_KEY_DEFAULT")
    if not api_key:
        parser.error(
            "API key must be provided via --api-key argument or "
            "OPENAI_API_KEY_DEFAULT environment variable"
        )

    # Process the files
    process_mp3_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        speaker_reference_path=args.speaker_reference,
        api_key=api_key,
    )

    print("\n✓ All files processed successfully")


if __name__ == "__main__":
    main()
