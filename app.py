import os
from threading import Thread

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# App Title & Layout
st.set_page_config(page_title="Podcaster GPT", page_icon="🎙️", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ Podcaster GPT")
st.markdown("Fine-tuned Gemma-3 for simulating podcasters. Ask anything!")

# Constants & Paths
PATH_WEIGHTS_BASE = "/Users/justinpyron/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
ADAPTER_ROOT = "/Users/justinpyron/code/podcaster-gpt/weights_dpo"

# Helper to find adapters
def get_adapters():
    if not os.path.exists(ADAPTER_ROOT):
        return []
    return [
        f
        for f in os.listdir(ADAPTER_ROOT)
        if os.path.isdir(os.path.join(ADAPTER_ROOT, f))
    ]


# --- Sidebar Configuration ---
st.sidebar.title("Configuration")

# About
st.sidebar.info(
    "This app uses a fine-tuned LoRA model based on Gemma-3 270m "
    "to simulate conversation with podcasters."
)

# Adapter Selection
available_adapters = get_adapters()
selected_adapter_name = st.sidebar.selectbox(
    "Select Adapter", available_adapters, index=0 if available_adapters else None
)

# Generation Params
max_new_tokens = st.sidebar.slider("Max New Tokens", 1, 512, 128)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model_and_tokenizer(adapter_name):
    st.info(f"Loading model with adapter: {adapter_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PATH_WEIGHTS_BASE)

    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load base model
    # Use torch_dtype for efficiency
    base_model = AutoModelForCausalLM.from_pretrained(
        PATH_WEIGHTS_BASE,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
    )
    base_model.eval()

    # Load adapter
    adapter_path = os.path.join(ADAPTER_ROOT, adapter_name)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        adapter_name=adapter_name,
    )

    return model, tokenizer, device


# Only load if we have an adapter selected
if selected_adapter_name:
    model, tokenizer, device = load_model_and_tokenizer(selected_adapter_name)
else:
    st.error(f"No adapters found in {ADAPTER_ROOT} directory.")
    st.stop()

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's on your mind?"):
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

        # Safely move inputs to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)

        # Set up streamer
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation arguments
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
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
