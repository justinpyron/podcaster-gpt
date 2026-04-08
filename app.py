import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL")


def stream_api(
    adapter_name: str,
    messages: list[dict],
    temperature: float = 1.0,
):
    """Stream generated text from the backend as token chunks."""
    if not BACKEND_URL:
        raise ValueError("BACKEND_URL environment variable is not set")

    with httpx.stream(
        "POST",
        f"{BACKEND_URL}/generate/{adapter_name}",
        json={
            "messages": messages,
            "temperature": temperature,
            "num_tokens": 256,
        },
        timeout=300.0,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            yield chunk


def _ping_backend():
    httpx.get(f"{BACKEND_URL}/health", timeout=120)


@st.fragment(run_every=2)
def backend_status():
    if st.session_state.get("backend_ready"):
        return
    st.info("Waking up the server. This may take a few seconds...", icon="⏳")
    if st.session_state.warmup_future.done():
        st.session_state.backend_ready = True
        st.rerun(scope="fragment")


st.set_page_config(page_title="Podcaster GPT", page_icon="🎙️", layout="centered")

if "warmup_future" not in st.session_state:
    executor = ThreadPoolExecutor(max_workers=1)
    st.session_state.warmup_future = executor.submit(_ping_backend)
    st.session_state.backend_ready = False

st.title("🎙️ Podcaster GPT")

backend_status()

temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

if st.button("Clear Chat"):
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
                adapter_name="base",
                messages=st.session_state.messages,
                temperature=temperature,
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": full_response})
