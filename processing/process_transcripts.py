"""
Script to process transcripts and generate formatted messages for training.
"""


def drop_first_and_last(messages: list[dict]) -> list[dict]:
    """
    Return a new list with the first and last messages removed. This is to discard
    clips that may be incomplete because they crossed a chunk boundary.

    Args:
        messages: List of dicts, each representing a message.

    Returns:
        List of dicts with first and last messages dropped. If the list has 2 or fewer messages, returns an empty list.
    """
    return messages[1:-1]


def merge_adjacent_speakers(
    messages: list[dict],
    threshold_seconds: float = 1,
    separator: str = "\n",
) -> list[dict]:
    """
    Merge adjacent messages that have the same speaker.
    If the time between messages is >= threshold_seconds seconds, separate by the given separator; otherwise, concatenate directly.

    Args:
        messages: List of dicts with keys: speaker, text, start, end. This is returned by the OpenAI transcription API.
        threshold_seconds: float, threshold in seconds to insert the separator between merged messages
        separator: str, string inserted between merged messages when the time gap is >= threshold

    Returns:
        List of dicts with the same format, with adjacent same-speaker messages merged
    """
    if not messages:
        return []

    merged = []
    current = messages[0].copy()

    for i in range(1, len(messages)):
        next_seg = messages[i]
        if next_seg["speaker"] == current["speaker"]:
            time_gap = next_seg["start"] - current["end"]
            if time_gap < threshold_seconds:
                current["text"] += next_seg["text"]
            else:
                current["text"] += separator + next_seg["text"]
            current["end"] = next_seg["end"]
        else:
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)

    return merged


def clean_messages(messages: list[dict]) -> list[dict]:
    """
    Clean messages by removing leading and trailing whitespace from the 'text' field.

    Args:
        messages: List of dicts, each representing a message.

    Returns:
        List of dicts with messages that have been stripped of leading and trailing whitespace in their 'text' field.
    """
    return [{**m, "text": m["text"].strip()} for m in messages]


def relabel_non_podcasters_as_guest(messages: list[dict]) -> list[dict]:
    """
    Relabel the 'speaker' value of all messages to 'guest' if it is not equal to 'podcaster'.

    Args:
        messages: List of dicts, each representing a message.

    Returns:
        List of dicts with the 'speaker' field set to 'guest' if it was not 'podcaster'.
    """
    return [
        {**m, "speaker": "guest" if m["speaker"] != "podcaster" else "podcaster"}
        for m in messages
    ]


def transform_into_chatbot_format(messages: list[dict]) -> list[dict]:
    """
    Transform the messages into the chatbot format.

    Args:
        messages: List of dicts, each representing a message.

    Returns:
        List of dicts with the messages transformed into the chatbot format.
    """
    return [
        {
            "role": "assistant" if m["speaker"] == "podcaster" else "user",
            "content": m["text"],
        }
        for m in messages
    ]


def process_transcript(messages: list[dict]) -> list[dict]:
    """
    Process the transcript and generate formatted messages for training.

    Args:
        messages: List of dicts, each representing a message.

    Returns:
        List of dicts with the messages processed for training.
    """
    messages = drop_first_and_last(messages)
    messages = relabel_non_podcasters_as_guest(messages)
    messages = merge_adjacent_speakers(messages, separator="\n\n")
    messages = clean_messages(messages)
    messages = transform_into_chatbot_format(messages)
    return messages
