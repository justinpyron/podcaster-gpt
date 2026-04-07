#!/usr/bin/env python3
"""
Clean processed transcripts using an LLM.

This is step 3b in the pipeline:
1. step1_chunk_mp3s.py: Split MP3 files into overlapping chunks
2. step2_transcribe.py: Convert MP3 files to raw transcript JSONs
3. step3_process_transcripts.py: Convert raw transcript JSONs to processed transcript JSONs
3b. step3b_clean_transcripts.py: Clean processed transcripts using an LLM
4. step4_create_sft_examples.py: Convert processed transcripts to SFT examples
5. step5_create_dpo_examples.py: Generate rejected completions for DPO training data
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from data_types import Message, ProcessedTranscript
from jinja2 import Template
from openai import OpenAI

# The cleaned example here comes from file rogan_2314__00.json
TEMPLATE = """
You are a transcript editor. Your job is to take a raw, choppy podcast transcript and produce a clean version that reads naturally.

## Problem

Automated transcription tools often produce fragmented output. When speakers talk near each other in time or there is cross-talk, the transcript rapidly alternates between speakers, splitting single thoughts across many tiny chunks. The result is choppy and hard to read.

## Your Task

Given a raw transcript as a JSON array of {"role", "content"} messages, produce a cleaned version in the same JSON format. The raw transcript is between <raw_transcript> and </raw_transcript> tags below.

Specifically:

1. **Merge fragmented turns.** When the same speaker has consecutive or near-consecutive chunks that are clearly part of one thought, combine them into a single message. This is the primary operation.

2. **Resolve cross-talk splicing.** When Speaker A's sentence is interrupted by a tiny fragment from Speaker B (like "yeah", "right", "mm-hmm") and then Speaker A continues the same thought, merge Speaker A's parts together. Keep Speaker B's backchannel acknowledgments only if they are substantive; drop pure filler interjections that were just artifacts of overlapping speech.

3. **Light touch only.** Preserve the speaker's original words as closely as possible. You may:
   - Join fragments with minimal bridging (e.g., removing a repeated word at a splice point)
   - Drop filler interjections from the other speaker that interrupt a thought ("yeah", "uh-huh", "right" when used as backchannel)
   - Fix obviously broken sentence boundaries caused by the chunking

   You may NOT:
   - Paraphrase or rewrite sentences
   - Add new ideas or words beyond trivial conjunctions at splice points
   - Remove substantive dialogue from any speaker
   - Change the meaning or emphasis of anything said

4. **Preserve conversation structure.** The output should still feel like a real conversation with natural back-and-forth. Don't over-merge — if a speaker genuinely pauses and the other person speaks, those should remain separate turns.

5. **Output valid JSON** in the same format: a JSON array of {"role": "user" | "assistant", "content": "..."} objects. Return only the cleaned JSON array, no code fences, no backticks, no commentary.

## Example 1

### Input (raw, choppy)
```
[
  {"role": "user", "content": "So the thing about distributed systems is"},
  {"role": "assistant", "content": "Right"},
  {"role": "user", "content": "you really can't avoid the CAP theorem, like it's"},
  {"role": "assistant", "content": "Yeah"},
  {"role": "user", "content": "fundamentally about tradeoffs between consistency and availability"},
  {"role": "assistant", "content": "And that's what people get wrong, they think they can somehow"},
  {"role": "user", "content": "Exactly"},
  {"role": "assistant", "content": "get around it with enough engineering, but you can't. You just have to choose which failure mode you prefer."},
  {"role": "user", "content": "Right, and most teams don't even realize they've made that choice implicitly."}
]
```

### Output (cleaned)
```
[
  {"role": "user", "content": "So the thing about distributed systems is you really can't avoid the CAP theorem, like it's fundamentally about tradeoffs between consistency and availability."},
  {"role": "assistant", "content": "And that's what people get wrong, they think they can somehow get around it with enough engineering, but you can't. You just have to choose which failure mode you prefer."},
  {"role": "user", "content": "Right, and most teams don't even realize they've made that choice implicitly."}
]
```

Notice: The user's fragmented thought was merged. The assistant's backchannel ("Right", "Yeah") was dropped. The assistant's split sentence was merged. The user's substantive "Exactly" was dropped since it was a minor interjection mid-thought, but the user's final substantive reply was kept. The original wording is almost entirely preserved.

## Example 2

### Input (raw, choppy)
```
[
    {
        "role": "assistant",
        "content": "All right. Now,"
    },
    {
        "role": "user",
        "content": "Hey,"
    },
    {
        "role": "assistant",
        "content": "what's"
    },
    {
        "role": "user",
        "content": "hey, oh,"
    },
    {
        "role": "assistant",
        "content": "happening?"
    },
    {
        "role": "user",
        "content": "lots happening."
    },
    {
        "role": "assistant",
        "content": "A"
    },
    {
        "role": "user",
        "content": "Really?"
    },
    {
        "role": "assistant",
        "content": "lot, a lot's"
    },
    {
        "role": "user",
        "content": "A lot."
    },
    {
        "role": "assistant",
        "content": "happening."
    },
    {
        "role": "user",
        "content": "A lot."
    },
    {
        "role": "assistant",
        "content": "Thank you so much for being here. I'm very excited to talk to you. I've been thinking about nothing but that since that dinner that we had a few months ago."
    },
    {
        "role": "user",
        "content": "Oh, yes."
    },
    {
        "role": "assistant",
        "content": "Thinking about it a lot."
    },
    {
        "role": "user",
        "content": "Yeah."
    },
    {
        "role": "assistant",
        "content": "You told me a lot of crazy stuff. So it's"
    },
    {
        "role": "user",
        "content": "Yeah,"
    },
    {
        "role": "assistant",
        "content": "like,"
    },
    {
        "role": "user",
        "content": "well,"
    },
    {
        "role": "assistant",
        "content": "when"
    },
    {
        "role": "user",
        "content": "it just seems like that's been my thing in life is get involved in the crazy stuff no matter where it comes from."
    },
    {
        "role": "assistant",
        "content": "did that start? When did you start getting involved in the crazy stuff?"
    },
    {
        "role": "user",
        "content": "Well, actually, I began early on. I was, you know, a ham radio operator as a teenager, and I went to vocational school. I didn't think I'd ever go to college or whatever, but I got all involved in... learning about radio transmission and all that kind of stuff. So I find this, I'm okay, I'm going to go to college and really concentrate on electrical engineering and physics and all that kind of stuff.\n\n But the weird stuff actually began kind of by absolute accident. At the time I was involved at Stanford University getting my PhD. I was just doing cool things. I'd invented a broadly tunable infrared laser, one of the first of its kind, even got a patent as a graduate student and co-authored with my thesis advisor a textbook.\n\n graduate-level textbook, Fundamentals of Quantum Electronics, published in English, French, Russian and Chinese. So I was on a cool roll just doing the normal physics kinds of things. But interestingly enough, once I was there writing a graduate-level textbook, I realized, you know, there's something I don't know, and that is... What about consciousness? What about living things? I mean, is it still just atoms and molecules all the way down? We just don't know about it? Or are there some additional fields or whatever? So it turned out I came across some publications by a polygraph expert who taught polygraph to the CIA and FBI and so on. And one day on a lark, he connected his polygraph up to his plants.\n\n And he saw signals coming out that looked like what you see out of people, and then he decided to threaten the plant like he would a person, and he got a big response."
    },
    {
        "role": "assistant",
        "content": "Whoa"
    },
    {
        "role": "user",
        "content": "And so he then went on to connect up a couple of plants to polygraphs, and he would find that if he affected one, the other one would respond. So I thought, okay, well maybe this is some new fields that we don't include in our physics. So I came up with what for me was just a pure physics experiment. experiment.\n\n I was going to grow some algae culture, split it up, put half of it at a laser link site far away and zap the local culture and see if it responded. I could measure velocity propagation and so on. So I sent that off to this polygraph guy, Cleve Baxter is his name, and so he said, well that'd be a cool experiment. Well here's one of these things where your life takes a left-hand turn.\n\n Totally random, he goes to a cocktail party in New York City, and there he runs into Ingo Swann, who turned out to be so-called psychic, famous artist, but fellow that did remote viewings, so-called.\n\n And so he invited him over to his lab and said see if he could affect the plants and so on. While he was there he saw my write-up about the experiment I proposed, which for me is just a pure physics experiment. And so he then wrote me a letter and said, well, if you're interested in the borderline between animate and inanimate physics, why deal with algae culture? They can't tell you anything. He should be dealing with somebody like me. Wow. I mean, I couldn't care less about dealing with, quote, a psychic or whatever. But attached to his letter, he had a big report that had been generated at City College in New York where he'd done some experiments where he would raise and lower the temperatures of sensitive temperature measuring devices across the lab. So I read that and I said, well, that's pretty interesting. So just on a lark. By this time I'd headed over to Stanford Research Institute to do my laser work.\n\n So anyway, I invited him for, you know, a weekend just to see what else he could do. And, of course, I talked to all my physics colleagues and said, oh, my God, these guys are all frauds and charlatans. You better know what you're doing. Well, it turns out that I had a great experiment for him because we had an experiment set up at Stanford that was a very sensitive quantum chip inside of electrical shielding, inside of magnetic shielding. Inside a superconducting shielding, completely acoustically isolated from the environment, no way anything on the outside could affect that little chip. They're only looking for quarks and stuff like that. So anyway, I brought him over to the lab. I said, remember that thing you did with the thermistors there at City College in New York? Well, this is sort of that like that on steroids. And so he said, okay, well, I'll see what I can do. Well, it turned out he generated all kinds of signals in that little quantum chip. And of course, a graduate student whose life depended on this not being, you know, affected by anything outside said, wait a minute. maybe there's some bubbles in the hydrogen line or something something but no he was able to do it but what was most interesting was and I asked him well how'd you know what to do he said well I didn't know what to do so I just looked inside looked inside through all this shielding and and he drew a diagram of what was inside there that never been published and he said well this is when I put my attention on it that just happened by accident"
    },
    {
        "role": "assistant",
        "content": "So he drew an accurate diagram of"
    },
    {
        "role": "user",
        "content": "an"
    },
    {
        "role": "assistant",
        "content": "all"
    },
    {
        "role": "user",
        "content": "accurate"
    },
    {
        "role": "assistant",
        "content": "the shielding that you had around this equipment."
    },
    {
        "role": "user",
        "content": "and the little quantum chip and its circuitry deep inside"
    },
    {
        "role": "assistant",
        "content": "And when you say he was able to affect something, what in particular was he able to affect? Perfect."
    },
    {
        "role": "user",
        "content": "Well in general there was a big oscillating signal coming out of the thing that ran about 30 seconds or so and then when he affected it it just stopped oscillating.\n\n And then he said, do you want me to do something else? And then he made it oscillate fast. And that's when the graduate students sort of went berserko.\n\n And so he said, wait, wait, let me see what's wrong here. And he couldn't find anything wrong. So he said, well, I'm sure that was just some kind of coincidental glitch. And he did it again.\n\n And so he said,"
    },
    {
        "role": "assistant",
        "content": "So"
    },
    {
        "role": "user",
        "content": "okay,"
    },
    {
        "role": "assistant",
        "content": "he's doing it exactly when he's saying he's going to do it."
    },
    {
        "role": "user",
        "content": "exactly when he says he's going to do it. But anyway, the reason I'm trying to get around to answering your question was that I then wrote this up and circulated it around to other physicists, and pretty soon the CIA come landing on my doorstep and said, oh, have we been looking for you. And I said, you know, why? Well, they looked in my background. They saw that I had, between my master's degree and Ph.D., I'd been a naval intelligence officer at the National Security Agency. I had lots of high-level clearances.\n\n And he said, you know, we have a problem. And they plopped a big report down on the desk about like that and said, look, the Russians have been spending millions of dollars at their best institutes trying to use ESP for espionage purposes. And we don't know how to evaluate it. I mean, no scientist in America even believes there is such a thing.\n\n And yet you did this experiment and it looked like this guy could actually get inside this device and describe it and affect it. And here you're at SRI. We have lots of black projects here anyway. So we'd like to check him out. Can you bring him back and let us come and do some experiments with him? And by the way, we're hoping that we'll find this is just all BS and we don't have to think about it and that'll be the end of that. So anyway, brought him back. They spent a day hiding things in the boxes and envelopes and he would describe what was inside and they were totally blown away. So they said okay we want would like to give you a little project here you know 50 or 60k and see what else he can do. So anyway that's how I got started on doing quote weird stuff and so as that many would know that project ended up being very productive and it went over.\n\n more than 20 years and so on, highly classified level and well maybe we'll s get to that separately because I think the U_A_P_ stuff is kind of more interesting to to start with, but anyway that's ha that's how I got started in in weird physics, you might call it, and then uh sort of like in Ghostbusters, well if you got some difficult problem who you going to call Al put off there I am\n\n Football's over, but the sports calendar doesn't slow down. With DraftKings Predictions, you can make real money predictions on sports, including pro basketball, college hoops, hockey, and other winter sports. There's always something to predict. Pick a topic, make a prediction. Will your team cover the spread? Will a college contender make a run?"
    }
]
```

### Output (cleaned)
```
[
    {
        "role": "assistant",
        "content": "All right. Now, what's happening?"
    },
    {
        "role": "user",
        "content": "Hey, oh, lots happening."
    },
    {
        "role": "assistant",
        "content": "A lot, a lot's happening."
    },
    {
        "role": "user",
        "content": "Really? A lot. A lot."
    },
    {
        "role": "assistant",
        "content": "Thank you so much for being here. I'm very excited to talk to you. I've been thinking about nothing but that since that dinner that we had a few months ago."
    },
    {
        "role": "user",
        "content": "Oh, yes."
    },
    {
        "role": "assistant",
        "content": "Thinking about it a lot."
    },
    {
        "role": "user",
        "content": "Yeah."
    },
    {
        "role": "assistant",
        "content": "You told me a lot of crazy stuff. So it's like,"
    },
    {
        "role": "user",
        "content": "Yeah, well, it just seems like that's been my thing in life is get involved in the crazy stuff no matter where it comes from."
    },
    {
        "role": "assistant",
        "content": "When did that start? When did you start getting involved in the crazy stuff?"
    },
    {
        "role": "user",
        "content": "Well, actually, I began early on. I was, you know, a ham radio operator as a teenager, and I went to vocational school. I didn't think I'd ever go to college or whatever, but I got all involved in... learning about radio transmission and all that kind of stuff. So I find this, I'm okay, I'm going to go to college and really concentrate on electrical engineering and physics and all that kind of stuff.\n\n But the weird stuff actually began kind of by absolute accident. At the time I was involved at Stanford University getting my PhD. I was just doing cool things. I'd invented a broadly tunable infrared laser, one of the first of its kind, even got a patent as a graduate student and co-authored with my thesis advisor a textbook.\n\n graduate-level textbook, Fundamentals of Quantum Electronics, published in English, French, Russian and Chinese. So I was on a cool roll just doing the normal physics kinds of things. But interestingly enough, once I was there writing a graduate-level textbook, I realized, you know, there's something I don't know, and that is... What about consciousness? What about living things? I mean, is it still just atoms and molecules all the way down? We just don't know about it? Or are there some additional fields or whatever? So it turned out I came across some publications by a polygraph expert who taught polygraph to the CIA and FBI and so on. And one day on a lark, he connected his polygraph up to his plants.\n\n And he saw signals coming out that looked like what you see out of people, and then he decided to threaten the plant like he would a person, and he got a big response."
    },
    {
        "role": "assistant",
        "content": "Whoa"
    },
    {
        "role": "user",
        "content": "And so he then went on to connect up a couple of plants to polygraphs, and he would find that if he affected one, the other one would respond. So I thought, okay, well maybe this is some new fields that we don't include in our physics. So I came up with what for me was just a pure physics experiment. experiment.\n\n I was going to grow some algae culture, split it up, put half of it at a laser link site far away and zap the local culture and see if it responded. I could measure velocity propagation and so on. So I sent that off to this polygraph guy, Cleve Baxter is his name, and so he said, well that'd be a cool experiment. Well here's one of these things where your life takes a left-hand turn.\n\n Totally random, he goes to a cocktail party in New York City, and there he runs into Ingo Swann, who turned out to be so-called psychic, famous artist, but fellow that did remote viewings, so-called.\n\n And so he invited him over to his lab and said see if he could affect the plants and so on. While he was there he saw my write-up about the experiment I proposed, which for me is just a pure physics experiment. And so he then wrote me a letter and said, well, if you're interested in the borderline between animate and inanimate physics, why deal with algae culture? They can't tell you anything. He should be dealing with somebody like me. Wow. I mean, I couldn't care less about dealing with, quote, a psychic or whatever. But attached to his letter, he had a big report that had been generated at City College in New York where he'd done some experiments where he would raise and lower the temperatures of sensitive temperature measuring devices across the lab. So I read that and I said, well, that's pretty interesting. So just on a lark. By this time I'd headed over to Stanford Research Institute to do my laser work.\n\n So anyway, I invited him for, you know, a weekend just to see what else he could do. And, of course, I talked to all my physics colleagues and said, oh, my God, these guys are all frauds and charlatans. You better know what you're doing. Well, it turns out that I had a great experiment for him because we had an experiment set up at Stanford that was a very sensitive quantum chip inside of electrical shielding, inside of magnetic shielding. Inside a superconducting shielding, completely acoustically isolated from the environment, no way anything on the outside could affect that little chip. They're only looking for quarks and stuff like that. So anyway, I brought him over to the lab. I said, remember that thing you did with the thermistors there at City College in New York? Well, this is sort of that like that on steroids. And so he said, okay, well, I'll see what I can do. Well, it turned out he generated all kinds of signals in that little quantum chip. And of course, a graduate student whose life depended on this not being, you know, affected by anything outside said, wait a minute. maybe there's some bubbles in the hydrogen line or something something but no he was able to do it but what was most interesting was and I asked him well how'd you know what to do he said well I didn't know what to do so I just looked inside looked inside through all this shielding and and he drew a diagram of what was inside there that never been published and he said well this is when I put my attention on it that just happened by accident"
    },
    {
        "role": "assistant",
        "content": "So he drew an accurate diagram of all the shielding that you had around this equipment."
    },
    {
        "role": "user",
        "content": "and the little quantum chip and its circuitry deep inside"
    },
    {
        "role": "assistant",
        "content": "And when you say he was able to affect something, what in particular was he able to affect? Perfect."
    },
    {
        "role": "user",
        "content": "Well in general there was a big oscillating signal coming out of the thing that ran about 30 seconds or so and then when he affected it it just stopped oscillating.\n\n And then he said, do you want me to do something else? And then he made it oscillate fast. And that's when the graduate students sort of went berserko.\n\n And so he said, wait, wait, let me see what's wrong here. And he couldn't find anything wrong. So he said, well, I'm sure that was just some kind of coincidental glitch. And he did it again.\n\n And so he said,"
    },
    {
        "role": "assistant",
        "content": "So he's doing it exactly when he's saying he's going to do it."
    },
    {
        "role": "user",
        "content": "okay, exactly when he says he's going to do it. But anyway, the reason I'm trying to get around to answering your question was that I then wrote this up and circulated it around to other physicists, and pretty soon the CIA come landing on my doorstep and said, oh, have we been looking for you. And I said, you know, why? Well, they looked in my background. They saw that I had, between my master's degree and Ph.D., I'd been a naval intelligence officer at the National Security Agency. I had lots of high-level clearances.\n\n And he said, you know, we have a problem. And they plopped a big report down on the desk about like that and said, look, the Russians have been spending millions of dollars at their best institutes trying to use ESP for espionage purposes. And we don't know how to evaluate it. I mean, no scientist in America even believes there is such a thing.\n\n And yet you did this experiment and it looked like this guy could actually get inside this device and describe it and affect it. And here you're at SRI. We have lots of black projects here anyway. So we'd like to check him out. Can you bring him back and let us come and do some experiments with him? And by the way, we're hoping that we'll find this is just all BS and we don't have to think about it and that'll be the end of that. So anyway, brought him back. They spent a day hiding things in the boxes and envelopes and he would describe what was inside and they were totally blown away. So they said okay we want would like to give you a little project here you know 50 or 60k and see what else he can do. So anyway that's how I got started on doing quote weird stuff and so as that many would know that project ended up being very productive and it went over.\n\n more than 20 years and so on, highly classified level and well maybe we'll s get to that separately because I think the U_A_P_ stuff is kind of more interesting to to start with, but anyway that's ha that's how I got started in in weird physics, you might call it, and then uh sort of like in Ghostbusters, well if you got some difficult problem who you going to call Al put off there I am\n\n Football's over, but the sports calendar doesn't slow down. With DraftKings Predictions, you can make real money predictions on sports, including pro basketball, college hoops, hockey, and other winter sports. There's always something to predict. Pick a topic, make a prediction. Will your team cover the spread? Will a college contender make a run?"
    }
]
```

## Reminder of the Task

Given a JSON array transcript, merge consecutive or interrupted same-speaker fragments into full thoughts.
1. Remove non-substantive backchannel interjections (e.g., "yeah", "right") when they break up another's sentence, but keep substantive ones.
2. Join fragments with minimal change; preserve original words and sentence structure—do not paraphrase or remove real content.
3. Fix obviously broken sentence boundaries if caused by chunking.
4. Maintain natural conversation turns—don’t over-merge if both speakers truly alternate.
5. Output only cleaned JSON array in the original [{"role", "content"}] format; nothing else.

## Raw transcript

<raw_transcript>
{{ raw_transcript }}
</raw_transcript>
"""


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
OPENAI_API_KEY_VAR = "OPENAI_API_KEY_DEFAULT"


def clean_transcript(
    messages: ProcessedTranscript,
    client: OpenAI,
    model: str = "gpt-5.4-nano",
    reasoning_effort: ReasoningEffort = "medium",
) -> ProcessedTranscript:
    """
    Clean a processed transcript by passing it through an LLM.

    Args:
        messages: ProcessedTranscript (list of Message).
        client: OpenAI client instance.
        model: Model to use.
        reasoning_effort: Reasoning effort level.

    Returns:
        ProcessedTranscript with cleaned messages.
    """
    prompt = Template(TEMPLATE).render(
        raw_transcript=json.dumps([m.model_dump() for m in messages])
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort},
    )
    cleaned = json.loads(response.output_text)
    return [Message.model_validate(m) for m in cleaned]


def process_single_file(
    json_file: Path,
    output_dir: Path,
    client: OpenAI,
    model: str,
    reasoning_effort: ReasoningEffort,
) -> tuple[Path, int | None, str | None]:
    """
    Process a single transcript file.

    Args:
        json_file: Path to the input JSON file.
        output_dir: Directory where cleaned transcript will be saved.
        client: OpenAI client instance.
        model: Model to use.
        reasoning_effort: Reasoning effort level.

    Returns:
        Tuple of (json_file, message_count, error_message).
        message_count is None if error occurred, error_message is None if successful.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        messages = [Message.model_validate(m) for m in raw_data]
        cleaned = clean_transcript(messages, client, model, reasoning_effort)

        output_path = output_dir / json_file.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [m.model_dump() for m in cleaned],
                f,
                indent=4,
                ensure_ascii=False,
            )

        return (json_file, len(cleaned), None)

    except Exception as e:
        return (json_file, None, str(e))


def process_all_files(
    input_dir: Path,
    output_dir: Path,
    model: str,
    reasoning_effort: ReasoningEffort,
    max_workers: int,
) -> None:
    """
    Clean all processed transcript JSON files in the input directory.

    Args:
        input_dir: Directory containing processed transcript JSON files.
        output_dir: Directory where cleaned transcript JSON files will be saved.
        model: OpenAI model to use.
        reasoning_effort: Reasoning effort level.
        max_workers: Maximum number of parallel workers.
    """
    api_key = os.getenv(OPENAI_API_KEY_VAR)
    if not api_key:
        raise ValueError(f"{OPENAI_API_KEY_VAR} environment variable is not set")

    client = OpenAI(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process")
    print(f"Model: {model}")
    print(f"Reasoning effort: {reasoning_effort}")
    print(f"Workers: {max_workers}\n")

    completed = 0
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                process_single_file,
                json_file,
                output_dir,
                client,
                model,
                reasoning_effort,
            ): json_file
            for json_file in json_files
        }

        for future in as_completed(future_to_file):
            completed += 1
            json_file, message_count, error = future.result()

            if error is None:
                successful += 1
                print(
                    f"[{completed}/{len(json_files)}] ✓ {json_file.name} "
                    f"({message_count} messages)"
                )
            else:
                failed += 1
                print(f"[{completed}/{len(json_files)}] ❌ {json_file.name}: {error}")

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"  Total: {len(json_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean processed transcripts using an LLM"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing processed transcript JSON files",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cleaned transcript JSON files will be saved",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-5.4-nano",
        help='OpenAI model to use (default: "gpt-5.4-nano")',
    )

    parser.add_argument(
        "-r",
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=REASONING_EFFORTS,
        help='Reasoning effort level (default: "medium")',
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=50,
        help="Maximum number of parallel workers (default: 50)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
