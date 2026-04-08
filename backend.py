"""Modal-based inference server for Podcaster GPT."""

import modal

# =============================================================================
# Configuration
# =============================================================================

VOLUME_NAME = "podcaster-gpt"
VOLUME_MOUNT_PATH = "/data"
MODEL_FOLDER_PATH = "gemma-3-1b-it"
ADAPTERS = {
    "rogan": "weights_sft/rogan-llm-cleaned-words5_20260407T215417Z",
    "dwarkesh": "weights_sft/dwarkesh-v0_20260408T164021Z",
}
SCALEDOWN_WINDOW_SECONDS = 200

# =============================================================================
# Modal Setup
# =============================================================================

app = modal.App("podcaster-gpt")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers",
    "peft",
    "fastapi",
    "pydantic",
)

volume = modal.Volume.from_name(VOLUME_NAME)


# =============================================================================
# Server
# =============================================================================


@app.cls(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
class Server:
    @modal.enter()
    def load_model_and_tokenizer(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = f"{VOLUME_MOUNT_PATH}/{MODEL_FOLDER_PATH}"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        base_model = AutoModelForCausalLM.from_pretrained(model_path)
        base_model.eval()

        adapter_names = sorted(ADAPTERS.keys())
        first = adapter_names[0]
        self.model = PeftModel.from_pretrained(
            base_model,
            f"{VOLUME_MOUNT_PATH}/{ADAPTERS[first]}",
            adapter_name=first,
        )
        for name in adapter_names[1:]:
            self.model.load_adapter(
                f"{VOLUME_MOUNT_PATH}/{ADAPTERS[name]}", adapter_name=name
            )

        self.model.eval()

    def generate(
        self,
        adapter_name: str,
        messages: list[dict],
        temperature: float,
        num_tokens: int,
    ):
        """Yield token chunks as the model generates a response."""
        import threading

        from transformers import TextIteratorStreamer

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=num_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        def _run():
            if adapter_name == "base":
                with self.model.disable_adapter():
                    self.model.generate(**generation_kwargs)
            else:
                self.model.set_adapter(adapter_name)
                self.model.generate(**generation_kwargs)

        thread = threading.Thread(target=_run)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel

        class Message(BaseModel):
            role: str
            content: str

        class GenerateRequest(BaseModel):
            messages: list[Message]
            temperature: float
            num_tokens: int

        api = FastAPI(title="Podcaster GPT API")

        @api.post("/generate/{adapter_name}")
        def generate_endpoint(
            adapter_name: str, request: GenerateRequest
        ) -> StreamingResponse:
            if adapter_name != "base" and adapter_name not in ADAPTERS:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Adapter '{adapter_name}' not found. "
                        f"Available: {sorted(ADAPTERS.keys())} or 'base'."
                    ),
                )

            return StreamingResponse(
                self.generate(
                    adapter_name=adapter_name,
                    messages=[m.model_dump() for m in request.messages],
                    temperature=request.temperature,
                    num_tokens=request.num_tokens,
                ),
                media_type="text/plain",
            )

        @api.get("/adapters")
        def list_adapters():
            return {"adapters": sorted(ADAPTERS.keys())}

        @api.get("/health")
        def health_check():
            return {"status": "healthy"}

        return api
