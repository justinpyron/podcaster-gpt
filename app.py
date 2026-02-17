import os
from threading import Thread

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- Configuration & Paths ---
PATH_WEIGHTS_BASE = "/Users/justinpyron/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
PATH_WEIGHTS_ADAPTER = "/Users/justinpyron/code/podcaster-gpt/weights_dpo/rogan-medium-words6_20260216T224542Z"

# --- App Setup ---
st.set_page_config(page_title="Podcaster GPT", page_icon="🎙️", layout="centered")

st.title("🎙️ Podcaster GPT")
st.markdown("Fine-tuned Gemma-3 for simulating podcasters.")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model_and_tokenizer():

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PATH_WEIGHTS_BASE)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(PATH_WEIGHTS_BASE)
    base_model.eval()

    # Load first adapter
    model = PeftModel.from_pretrained(
        base_model,
        PATH_WEIGHTS_ADAPTER,
        adapter_name="rogan-medium-words6",
    )
    model.eval()

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()

# --- Controls ---
# Keep it simple: just a temperature slider in the main view
temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare for assistant response
    with st.chat_message("assistant"):
        # Format messages for the chat template
        chat_history = st.session_state.messages

        # Tokenize with chat template
        inputs = tokenizer.apply_chat_template(
            chat_history,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )

        # Set up streamer
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation arguments
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=256,
            do_sample=True,
            temperature=temperature,
            # top_p=0.9,  # Stabilize generation by filtering low-prob tokens
            # top_k=50,   # Stabilize generation by filtering the tail
            pad_token_id=tokenizer.pad_token_id,
        )

        # Run generation in a separate thread to allow streaming
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Display response using stream
        def response_generator():
            for new_text in streamer:
                yield new_text

        full_response = st.write_stream(response_generator())

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
