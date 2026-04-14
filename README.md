# Podcaster GPT

Chat model fine-tuned to mimic famous podcasters.

## Overview

This project fine-tunes LoRA adapters on a base language model ([Gemma-3-1B-IT](https://huggingface.co/google/gemma-3-1b-it)) to emulate the conversational styles of podcast hosts. Training data is created by transcribing podcast episodes and formatting them into supervised fine-tuning (SFT) and direct preference optimization (DPO) datasets. Currently supported podcasters: **Joe Rogan** and **Dwarkesh Patel**.

## Architecture

- **Frontend:** Streamlit app deployed on Google Cloud Run
- **Backend:** FastAPI streaming inference server on Modal (GPU)
- **Training:** SFT and DPO with LoRA on Modal GPUs
- **Data pipeline:** Podcast MP3s → transcription (OpenAI API) → SFT/DPO dataset creation
- **CI/CD:** GitHub Actions for automated deployment

## Project Structure

```
├── app.py                                  # Streamlit frontend
├── backend.py                              # Modal FastAPI inference server
├── train_sft.py                            # SFT training on Modal
├── train_dpo.py                            # DPO training on Modal
├── processing/
│   ├── data_types.py                       # Pydantic models for the pipeline
│   ├── step1_chunk_mp3s.py                 # Split MP3s into overlapping chunks
│   ├── step2_transcribe.py                 # Transcribe audio via OpenAI API
│   ├── step3_process_transcripts.py        # Format raw transcripts into messages
│   ├── step4_create_sft_examples.py        # Create SFT prompt/completion pairs
│   ├── step5_create_dpo_examples.py        # Generate rejected completions for DPO
│   └── step6_train_val_split.py            # Split data into train/val sets
├── Dockerfile                              # Cloud Run container config
├── .github/workflows/build-and-deploy.yml  # CI/CD pipeline
└── pyproject.toml                          # uv dependencies
```

## Setup

1. Install [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:

```bash
uv sync
```

## Data Preparation

The data pipeline transforms raw podcast MP3s into training-ready datasets:

```
MP3s → chunks → transcripts → processed messages → SFT examples → DPO examples → train/val split
```

Run each step sequentially:

```bash
# 1. Split MP3s into overlapping chunks
python processing/step1_chunk_mp3s.py --input-dir <mp3s> --output-dir <chunks>

# 2. Transcribe chunks via OpenAI API
python processing/step2_transcribe.py --input-dir <chunks> --output-dir <raw-transcripts>

# 3. Process raw transcripts into formatted messages
python processing/step3_process_transcripts.py --input-dir <raw-transcripts> --output-dir <processed>

# 4. Create SFT examples
python processing/step4_create_sft_examples.py --input-dir <processed> --output-dir <sft-examples>

# 5. Generate DPO examples (rejected completions)
python processing/step5_create_dpo_examples.py --input-dir <sft-examples> --output-dir <dpo-examples>

# 6. Split into train/val sets
python processing/step6_train_val_split.py --input-dir <examples> --output-dir <split>
```

## Training

Train a LoRA adapter on Modal:

```bash
# SFT
modal run train_sft.py \
  --model-path gemma-3-1b-it \
  --data-path-train data/sft/rogan-split/train.json \
  --data-path-val data/sft/rogan-split/val.json \
  --name rogan-1b

# DPO
modal run train_dpo.py \
  --model-path gemma-3-1b-it \
  --data-path-train data/dpo/rogan-split/train.json \
  --data-path-val data/dpo/rogan-split/val.json \
  --name rogan-1b-dpo
```

Base model weights and datasets must be uploaded to the `podcaster-gpt` Modal volume before training. All `--model-path` and `--data-path-*` arguments are relative to the volume root. Training logs are sent to Weights & Biases.

## Deployment

Deploy backend and frontend via GitHub Actions (trigger manually from the GitHub UI). The workflow:

1. Deploys the Modal backend (`backend.py`)
2. Builds a Docker image for the Streamlit app
3. Deploys to Google Cloud Run with `BACKEND_URL` env var

## Local Development

Run the app locally (requires `BACKEND_URL` env var):

```bash
export BACKEND_URL="https://your-modal-backend.modal.run"
uv run streamlit run app.py
```
