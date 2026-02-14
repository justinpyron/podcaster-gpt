# Processing Pipeline

This folder contains the scripts for the 5-step data processing pipeline used to generate training data for the podcaster LLM.

## Pipeline Steps

1.  **`step1_chunk_mp3s.py`**
    Splits long audio files into smaller, overlapping MP3 chunks.
    *   **Input:** Raw MP3 files.
    *   **Output:** Overlapping MP3 chunks (default: 5-minute chunks with 20-second overlap).

2.  **`step2_transcribe.py`**
    Transcribes MP3 chunks into raw JSON transcripts using OpenAI's transcription API with speaker diarization.
    *   **Input:** MP3 chunks + Speaker reference audio (2-10s clip of the podcaster).
    *   **Output:** Raw JSON segments with speaker labels and timestamps.

3.  **`step3_process_transcripts.py`**
    Cleans and formats raw transcripts for training.
    *   **Processing:** Drops edge segments, merges adjacent same-speaker segments, and relabels speakers to `podcaster` and `guest`.
    *   **Output:** Formatted conversation JSONs (list of `Message` objects).

4.  **`step4_create_sft_examples.py`**
    Converts processed conversations into Supervised Fine-Tuning (SFT) examples.
    *   **Format:** Creates `(prompt, completion)` pairs for every podcaster response in the transcript.
    *   **Output:** JSON files containing `SFTExample` objects.

5.  **`step5_create_dpo_examples.py`**
    Generates Direct Preference Optimization (DPO) examples from SFT data.
    *   **Generation:** Uses a separate LLM (via Together AI) to generate "rejected" completions for the given prompts.
    *   **Output:** JSON files containing `DPOExample` objects with `prompt`, `chosen`, and `rejected` fields.

## Shared Utilities

*   **`data_types.py`**: Pydantic models (e.g., `Message`, `SFTExample`, `DPOExample`) used across the pipeline to ensure data consistency.
