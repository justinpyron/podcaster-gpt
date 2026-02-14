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


RawTranscript = list[RawTranscriptSegment]

ProcessedTranscript = list[ProcessedTranscriptMessage]
