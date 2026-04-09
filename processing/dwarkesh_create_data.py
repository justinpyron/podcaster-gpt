#!/usr/bin/env python3
"""
Scrape transcripts from Dwarkesh Patel's podcast and generate formatted messages for training.

This script iterates through a list of Dwarkesh podcast URLs, extracts the transcript,
and converts it into a list of Message objects (ProcessedTranscript), which can then
be used by step4_create_sft_examples.py.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from data_types import Message, ProcessedTranscript


def extract_html(url: str) -> BeautifulSoup | None:
    """
    Fetches the content of a URL and returns a BeautifulSoup object.

    Args:
        url: The URL to fetch.

    Returns:
        A BeautifulSoup object if the request was successful, None otherwise.
    """
    try:
        response = httpx.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except httpx.HTTPError as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_transcript(url: str) -> list[dict[str, str]]:
    """
    Given a URL, fetch the HTML with extract_html and parse the transcript section.

    Args:
        url: The URL of the Dwarkesh podcast episode.

    Returns:
        A list of dictionaries with 'speaker' and 'content' keys.
    """
    soup = extract_html(url)
    if soup is None:
        return []
    transcript = []
    current_speaker = None

    # 1. Find the specific "Transcript" anchor by its unique ID
    anchor = soup.find(id="§transcript")
    if not anchor:
        return []  # Return empty if transcript section isn't found

    # 2. Get the parent H2 so we can look at its siblings
    transcript_header = anchor.find_parent("h2")

    # 3. Only look at elements AFTER this header
    # find_next_siblings() returns a list of all tags following the header
    for element in transcript_header.find_next_siblings():
        # We only care about <p> tags for the conversation
        if element.name == "p":
            speaker_tag = element.find("strong")

            if speaker_tag:
                current_speaker = speaker_tag.get_text(strip=True)
                transcript.append({"speaker": current_speaker, "content": ""})
            elif current_speaker:
                text = element.get_text(" ", strip=True)
                if transcript and transcript[-1]["content"]:
                    transcript[-1]["content"] += "\n" + text
                elif transcript:
                    transcript[-1]["content"] = text

    return transcript


def transform_to_processed_transcript(
    raw_transcript: list[dict[str, str]],
    podcaster_name: str = "Dwarkesh",
) -> ProcessedTranscript:
    """
    Transform the raw scraped transcript into the ProcessedTranscript format.
    Dwarkesh Patel is mapped to 'assistant', and everyone else to 'user'.
    Consecutive messages from the same role are merged with two newlines.

    Args:
        raw_transcript: List of dicts with 'speaker' and 'content'.
        podcaster_name: The name of the podcaster to map to 'assistant'.

    Returns:
        ProcessedTranscript (list of Message).
    """
    if not raw_transcript:
        return []

    processed = []
    for turn in raw_transcript:
        role = (
            "assistant" if podcaster_name.lower() in turn["speaker"].lower() else "user"
        )
        content = turn["content"]

        if processed and processed[-1].role == role:
            processed[-1].content += "\n\n" + content
        else:
            processed.append(Message(role=role, content=content))

    return processed


def process_urls(urls: list[str], output_dir: Path) -> None:
    """
    Process a list of URLs, scrape transcripts, and save them as JSON files.

    Args:
        urls: List of Dwarkesh podcast URLs.
        output_dir: Directory to save the processed JSON files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(urls)} URL(s) to process\n")

    successful = 0
    failed = 0

    for url in urls:
        try:
            # Derive a filename from the URL
            slug = url.rstrip("/").split("/")[-1]
            output_path = output_dir / f"{slug}.json"

            print(f"Processing {url}...")
            raw_transcript = extract_transcript(url)

            if not raw_transcript:
                print(f"  ⚠️ No transcript found for {url}")
                failed += 1
                continue

            processed_transcript = transform_to_processed_transcript(raw_transcript)

            if not processed_transcript:
                print(f"  ⚠️ Processed transcript is empty for {url}")
                failed += 1
                continue

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [m.model_dump() for m in processed_transcript],
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            successful += 1
            print(
                f"  ✓ Saved to {output_path.name} ({len(processed_transcript)} turns)"
            )

        except Exception as e:
            failed += 1
            print(f"  ❌ Error processing {url}: {e}")

    print(f"\n{'='*60}")
    print("Scraping complete!")
    print(f"  Total URLs: {len(urls)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Scrape Dwarkesh transcripts and convert to training format"
    )

    parser.add_argument(
        "-u",
        "--urls",
        nargs="+",
        help="List of Dwarkesh podcast URLs to process",
    )

    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Path to a text file containing URLs (one per line)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where processed transcript JSON files will be saved",
    )

    args = parser.parse_args()

    urls = []
    if args.urls:
        urls.extend(args.urls)
    if args.file:
        if not args.file.exists():
            parser.error(f"URL file does not exist: {args.file}")
        with open(args.file, "r") as f:
            urls.extend([line.strip() for line in f if line.strip()])

    if not urls:
        parser.error("No URLs provided. Use --urls or --file.")

    process_urls(urls, args.output_dir)


if __name__ == "__main__":
    main()
