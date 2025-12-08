"""Minimal Modal deployment for dermatology LLM assistant."""

import os
import modal
from fastapi.responses import StreamingResponse
import json
import asyncio

# Configuration from environment variables
MODEL_NAME = os.getenv("MODAL_MODEL_NAME", "medgemma-27b")
MAX_TOKENS = int(os.getenv("MODAL_MAX_TOKENS", "700"))
GPU = os.getenv("MODAL_GPU", "H200")

# Build image
llm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch",
        "transformers[torch,vision]",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "bitsandbytes",
        "huggingface-hub==0.36.0",
        "fastapi[standard]",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("llm.py", remote_path="/root/llm.py")
    .add_local_file("prompts.py", remote_path="/root/prompts.py")
)

# Create app
secrets = [modal.Secret.from_name("hf")]
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App(
    name=f"dermatology-llm-{MODEL_NAME.split('-')[1]}",
    image=llm_image,
    secrets=secrets,
    volumes={"/root/.cache/huggingface": hf_cache},
)


@app.cls(gpu=GPU, min_containers=0, max_containers=1, scaledown_window=200)
class DermatologyLLM:
    """Modal class for dermatology LLM assistant."""

    @modal.enter()
    def initialize_model(self):
        """Initialize the LLM model."""
        import sys

        sys.path.insert(0, "/root")

        from llm import LLM
        from prompts import BASE_PROMPT, QUESTION_PROMPT, TIME_TRACKING_PROMPT

        print(f"Initializing {MODEL_NAME} with max_tokens={MAX_TOKENS}")

        self.llm = LLM(
            model_name=MODEL_NAME,
            max_new_tokens=MAX_TOKENS,
            base_prompt=BASE_PROMPT,
            question_prompt=QUESTION_PROMPT,
            time_tracking_prompt=TIME_TRACKING_PROMPT,
        )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def explain(self, json_data: dict) -> str:
        """Generate explanation from predictions and metadata."""
        return self.llm.explain(predictions=json_data["predictions"], metadata=json_data["metadata"], temperature=0.3)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def ask_followup(self, json_data: dict) -> dict:
        """Answer follow-up question with conversation history."""
        return self.llm.ask_followup(
            initial_answer=json_data["initial_answer"],
            question=json_data["question"],
            conversation_history=json_data.get("conversation_history", []),
            temperature=0.3,
        )

    @modal.fastapi_endpoint(method="POST")
    async def ask_followup_stream(self, json_data: dict):
        initial_answer = json_data["initial_answer"]
        question = json_data["question"]
        history = json_data.get("conversation_history", [])

        # Generator that yields bytes for HTTP streaming
        async def event_stream():
            for item in self.llm.ask_followup_stream(
                initial_answer=initial_answer,
                question=question,
                conversation_history=history,
            ):
                # Yield newline-delimited JSON (important!)
                yield (json.dumps(item) + "\n").encode("utf-8")
                await asyncio.sleep(0)  # allow event loop to flush

        return StreamingResponse(event_stream(), media_type="application/json")

    @modal.fastapi_endpoint(method="POST")
    async def explain_stream(self, json_data: dict):
        """
        Streaming endpoint for explanation output.
        Expects same format as non-streaming explain endpoint:
            {
                "predictions": {"disease1": 0.78, "disease2": 0.15, ...},
                "metadata": {"user_input": "...", "cv_analysis": {...}, "history": {...}}
            }

        Streams newline-delimited JSON objects:
            {"delta": "..."}\n
            {"delta": "..."}\n
            ...
        """
        predictions = json_data["predictions"]
        metadata = json_data["metadata"]

        # Build prompt from predictions and metadata (same as non-streaming version)
        prompt = self.llm.build_prompt(predictions, metadata)

        async def event_stream():
            # explain_stream_generator yields dicts of the form {"delta": "text"}
            for chunk in self.llm.explain_stream_generator(prompt):
                if chunk:
                    # Send as JSONL (chunk is already a dict with "delta" key)
                    yield (json.dumps(chunk) + "\n").encode("utf-8")
                # Yield control to event loop so client gets tokens immediately
                await asyncio.sleep(0)

        return StreamingResponse(event_stream(), media_type="application/json")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def time_tracking_summary(self, json_data: dict) -> dict:
        """
        Generate a brief summary of time tracking data for a skin condition.

        Expects:
            {
                "predictions": {"disease1": 0.78, ...},
                "user_demographics": {...},
                "user_input": "text description",
                "cv_analysis_history": {
                    "2024-12-01": {"area_cm2": 2.5, "color_stats_lab": {...}, ...},
                    "2024-12-08": {"area_cm2": 2.0, "color_stats_lab": {...}, ...}
                }
            }

        Returns:
            {"summary": "3-4 sentence summary text"}
        """
        return self.llm.time_tracking_summary(
            predictions=json_data["predictions"],
            user_demographics=json_data.get("user_demographics", {}),
            user_input=json_data.get("user_input", ""),
            cv_analysis_history=json_data["cv_analysis_history"],
            temperature=0.3,
        )
