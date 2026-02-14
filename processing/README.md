# Processing Pipeline

This folder contains the scripts for the 5-step data processing pipeline used to generate training data for the podcaster LLM.

## Pipeline Steps

1.  **`step1_chunk_mp3s.py`**
    Splits long audio files into smaller, overlapping MP3 chunks.
    *   **Input Folder:** `data/podcasts/` (Raw MP3 files)
    *   **Output Folder:** `data/chunks/` (Overlapping MP3 chunks)

2.  **`step2_transcribe.py`**
    Transcribes MP3 chunks into raw JSON transcripts using OpenAI's transcription API.
    *   **Input Folder:** `data/chunks/` (MP3 chunks)
    *   **Output Folder:** `data/transcripts_raw/` (Raw JSON with diarization)
    *   **Reference:** `data/speaker_samples/` (Podcaster reference audio)

3.  **`step3_process_transcripts.py`**
    Cleans raw transcripts and formats them into conversation messages.
    *   **Input Folder:** `data/transcripts_raw/` (Raw JSON)
    *   **Output Folder:** `data/transcripts_processed/` (Formatted messages)

4.  **`step4_create_sft_examples.py`**
    Converts processed conversations into Supervised Fine-Tuning (SFT) examples.
    *   **Input Folder:** `data/transcripts_processed/` (Formatted messages)
    *   **Output Folder:** `data/examples_sft/` (SFT examples JSON)

5.  **`step5_create_dpo_examples.py`**
    Generates Direct Preference Optimization (DPO) examples with rejected completions.
    *   **Input Folder:** `data/examples_sft/` (SFT examples)
    *   **Output Folder:** `data/examples_dpo/` (DPO examples JSON)

## Data Directory Structure

*   **`data/podcasts/`**: Full-length podcast MP3 files.
*   **`data/chunks/`**: The MP3 files split into chunks.
*   **`data/speaker_samples/`**: Short clips used to identify the podcaster during transcription.
*   **`data/transcripts_raw/`**: Initial transcripts with raw speaker labels and timestamps.
*   **`data/transcripts_processed/`**: Cleaned conversations relabeled for training.
*   **`data/examples_sft/`**: Training data formatted for Supervised Fine-Tuning.
*   **`data/examples_dpo/`**: Training data formatted for Direct Preference Optimization.

## Shared Utilities

*   **`data_types.py`**: Pydantic models (e.g., `Message`, `SFTExample`, `DPOExample`) used across the pipeline to ensure data consistency.

## Example Execution

Run the full pipeline sequentially from the project root using `uv`. This example processes the `rogan` podcast:

```bash
# 1. Chunk audio files
uv run processing/step1_chunk_mp3s.py \
    -i data/podcasts/rogan \
    -o data/chunks/rogan

# 2. Transcribe chunks (requires OpenAI API key)
uv run processing/step2_transcribe.py \
    -i data/chunks/rogan \
    -o data/transcripts_raw/rogan \
    -s data/speaker_samples/rogan.mp3

# 3. Clean and format transcripts
uv run processing/step3_process_transcripts.py \
    -i data/transcripts_raw/rogan \
    -o data/transcripts_processed/rogan

# 4. Create SFT examples
uv run processing/step4_create_sft_examples.py \
    -i data/transcripts_processed/rogan \
    -o data/examples_sft/rogan

# 5. Generate DPO examples (requires Together AI API key)
uv run processing/step5_create_dpo_examples.py \
    -i data/examples_sft/rogan \
    -o data/examples_dpo/rogan
```
