"""Minimal Modal deployment for dermatology LLM assistant."""
import os
import modal

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
        "fastapi[standard]"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    })
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
    volumes={"/root/.cache/huggingface": hf_cache}
)


@app.cls(
    gpu=GPU,
    min_containers=0,
    max_containers=1,
    scaledown_window=100
)
class DermatologyLLM:
    """Modal class for dermatology LLM assistant."""
    
    @modal.enter()
    def initialize_model(self):
        """Initialize the LLM model."""
        import sys
        sys.path.insert(0, '/root')
        
        from llm import LLM
        from prompts import BASE_PROMPT, QUESTION_PROMPT
        
        print(f"Initializing {MODEL_NAME} with max_tokens={MAX_TOKENS}")
        
        self.llm = LLM(
            model_name=MODEL_NAME,
            max_new_tokens=MAX_TOKENS,
            base_prompt=BASE_PROMPT,
            question_prompt=QUESTION_PROMPT
        )
    
    @modal.fastapi_endpoint(method="POST", docs=True)
    def explain(self, json_data: dict) -> str:
        """Generate explanation from predictions and metadata."""
        return self.llm.explain(
            predictions=json_data['predictions'],
            metadata=json_data['metadata'],
            temperature=0.3
        )
    
    @modal.fastapi_endpoint(method="POST", docs=True)
    def ask_followup(self, json_data: dict) -> dict:
        """Answer follow-up question with conversation history."""
        return self.llm.ask_followup(
            initial_answer=json_data['initial_answer'],
            question=json_data['question'],
            conversation_history=json_data.get('conversation_history', []),
            temperature=0.3
        )
