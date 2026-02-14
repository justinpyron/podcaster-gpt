"""
Pydantic models for transcript data objects used throughout the processing pipeline.
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
