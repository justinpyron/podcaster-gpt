#!/usr/bin/env python3
"""
Script to split MP3 files into chunks.

Iterates through MP3 files in an input folder, splits each into overlapping chunks,
and saves them to an output folder with a structured naming convention.
"""

import argparse
from pathlib import Path
from typing import List

from pydub import AudioSegment


def split_audio_to_chunks(
    input_path: Path,
    output_dir: Path,
    folder_name: str,
    chunk_length_ms: int = 5 * 60 * 1000,
    overlap_ms: int = 20 * 1000,
) -> List[Path]:
    """
    Splits an mp3 audio file into overlapping chunks and saves them as mp3 files.

    Args:
        input_path: Path to the input mp3 file.
        output_dir: Directory where chunked files will be saved.
        folder_name: Name of the input folder (for naming chunks).
        chunk_length_ms: Length of each chunk in milliseconds.
        overlap_ms: Overlap between consecutive chunks in milliseconds.

    Returns:
        List of paths to the created chunk files.
    """
    print(f"Loading audio file: {input_path}")
    audio = AudioSegment.from_mp3(input_path)

    chunks = []
    start = 0
    while start < len(audio):
        end = start + chunk_length_ms
        chunks.append(audio[start:end])
        start = end - overlap_ms  # Overlap for consecutive chunks

    # Get the input file name without extension
    input_file_stem = input_path.stem

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        # Format: pod_{folder_name}__pod_{file_name}__chunk_{chunk_idx}
        chunk_filename = f"pod_{folder_name}__pod_{input_file_stem}__chunk_{i:02d}.mp3"
        chunk_path = output_dir / chunk_filename

        print(f"  Exporting chunk {i+1:02d} of {len(chunks)}: {chunk_filename}")
        chunk.export(str(chunk_path), format="mp3")
        chunk_paths.append(chunk_path)

    return chunk_paths


def process_folder(
    input_folder: Path,
    output_folder: Path,
    chunk_length_ms: int,
    overlap_ms: int,
) -> None:
    """
    Process all MP3 files in the input folder.

    Args:
        input_folder: Path to the folder containing MP3 files.
        output_folder: Path to the folder where chunks will be saved.
        chunk_length_ms: Length of each chunk in milliseconds.
        overlap_ms: Overlap between consecutive chunks in milliseconds.
    """
    # Get the input folder name for naming chunks
    folder_name = input_folder.name

    # Find all MP3 files in the input folder
    mp3_files = sorted(input_folder.glob("*.mp3"))

    if not mp3_files:
        print(f"No MP3 files found in {input_folder}")
        return

    print(f"Found {len(mp3_files)} MP3 file(s) in {input_folder}")
    print(f"Chunk length: {chunk_length_ms / 1000}s, Overlap: {overlap_ms / 1000}s")
    print(f"Output directory: {output_folder}\n")

    for idx, mp3_file in enumerate(mp3_files, 1):
        print(f"[{idx}/{len(mp3_files)}] Processing: {mp3_file.name}")

        chunk_paths = split_audio_to_chunks(
            input_path=mp3_file,
            output_dir=output_folder,
            folder_name=folder_name,
            chunk_length_ms=chunk_length_ms,
            overlap_ms=overlap_ms,
        )

        print(f"  Created {len(chunk_paths)} chunk(s)\n")

    print(f"Processing complete! All chunks saved to {output_folder}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Split MP3 files into overlapping chunks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input-folder",
        type=Path,
        required=True,
        help="Path to the folder containing MP3 files to process",
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        type=Path,
        required=True,
        help="Path to the folder where chunks will be saved",
    )

    parser.add_argument(
        "-c",
        "--chunk-length",
        type=int,
        default=300,
        help="Length of each chunk in seconds (default: 300 = 5 minutes)",
    )

    parser.add_argument(
        "-l",
        "--overlap",
        type=int,
        default=20,
        help="Overlap between consecutive chunks in seconds (default: 20)",
    )

    args = parser.parse_args()

    # Validate input folder exists
    if not args.input_folder.exists():
        parser.error(f"Input folder does not exist: {args.input_folder}")

    if not args.input_folder.is_dir():
        parser.error(f"Input path is not a directory: {args.input_folder}")

    # Convert seconds to milliseconds
    chunk_length_ms = args.chunk_length * 1000
    overlap_ms = args.overlap * 1000

    # Process the folder
    process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        chunk_length_ms=chunk_length_ms,
        overlap_ms=overlap_ms,
    )


if __name__ == "__main__":
    main()
