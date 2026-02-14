"""
Pydantic models for transcript data objects used throughout the processing pipeline.
"""
from pydantic import BaseModel


class RawTranscriptSegment(BaseModel):
    """A single segment from the OpenAI transcription API output."""

    speaker: str
    text: str
    start: float
    end: float


class ProcessedTranscriptMessage(BaseModel):
    """A single message in chatbot training format."""

    role: str
    content: str


class SftExample(BaseModel):
    """A single SFT training example: prompt messages paired with the target completion."""

    prompt: list[ProcessedTranscriptMessage]
    completion: list[ProcessedTranscriptMessage]


RawTranscript = list[RawTranscriptSegment]

ProcessedTranscript = list[ProcessedTranscriptMessage]
