"""
Pydantic models for transcript data objects used throughout the processing pipeline.

The pipeline consists of the following steps:
1. step1_chunk_mp3s.py: Split MP3 files into overlapping chunks
2. step2_transcribe.py: Convert MP3 files to raw transcript JSONs
3. step3_process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
4. step4_create_sft_examples.py: Convert processed transcripts to SFT examples
5. step5_create_dpo_examples.py: Generate rejected completions for DPO training data
"""
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """A single segment from the OpenAI transcription API output."""

    speaker: str
    text: str
    start: float
    end: float


class Message(BaseModel):
    """A single message in chatbot training format."""

    role: str
    content: str


class SFTExample(BaseModel):
    """A single SFT training example: prompt messages paired with the target completion."""

    prompt: list[Message]
    completion: list[Message]


class DPOExample(BaseModel):
    """A single DPO training example: prompt, chosen completion, and rejected completion."""

    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]


RawTranscript = list[TranscriptSegment]

ProcessedTranscript = list[Message]
