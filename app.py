import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL")
NUM_TOKENS = 256
TEMPERATURE = 1.0
TIMEOUT_SECONDS = 300.0
# Keys in the dict below must match the adapter endpoint names in the backend.
PODCASTERS = {
    "base": "Base",
    "rogan": "Joe Rogan",
    "dwarkesh": "Dwarkesh Patel",
}
WHAT_IS_THIS_APP = """\
Chat with an AI that talks like your favorite podcasters.

This app uses LoRA adapters fine-tuned on podcast transcripts to capture each
host's distinctive conversational style. Select "Base" to chat with the
original model ([Gemma-3-1B-IT](https://huggingface.co/google/gemma-3-1b-it))
without any style tuning.

Source code 👉 [GitHub](https://github.com/justinpyron/podcaster-gpt)
"""


def stream_api(
    adapter_name: str,
    messages: list[dict],
):
    """Stream generated text from the backend as token chunks."""
    if not BACKEND_URL:
        raise ValueError("BACKEND_URL environment variable is not set")

    with httpx.stream(
        "POST",
        f"{BACKEND_URL}/generate/{adapter_name}",
        json={
            "messages": messages,
            "temperature": TEMPERATURE,
            "num_tokens": NUM_TOKENS,
        },
        timeout=TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            yield chunk


def _ping_backend():
    httpx.get(f"{BACKEND_URL}/health", timeout=300)


@st.fragment(run_every=2)
def backend_status():
    if st.session_state.get("backend_ready"):
        return
    st.info("Waking up the server. This may take a few seconds...", icon="⏳")
    if st.session_state.warmup_future.done():
        st.session_state.backend_ready = True


st.set_page_config(page_title="Podcaster GPT", page_icon="🎙️", layout="centered")

if "warmup_future" not in st.session_state:
    executor = ThreadPoolExecutor(max_workers=1)
    st.session_state.warmup_future = executor.submit(_ping_backend)
    st.session_state.backend_ready = False

st.title("🎙️ Podcaster GPT")
backend_status()

with st.expander("What is this app?"):
    st.markdown(WHAT_IS_THIS_APP)

selected_podcaster = st.segmented_control(
    "Podcaster",
    options=list(PODCASTERS.keys()),
    default=list(PODCASTERS.keys())[0],
    format_func=lambda x: PODCASTERS[x],
    width="stretch",
)

if "active_podcaster" not in st.session_state:
    st.session_state.active_podcaster = selected_podcaster
if selected_podcaster != st.session_state.active_podcaster:
    st.session_state.active_podcaster = selected_podcaster
    st.session_state.messages = []
    st.rerun()

if st.button("Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = st.write_stream(
            stream_api(
                adapter_name=selected_podcaster.lower(),
                messages=st.session_state.messages,
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": full_response})
