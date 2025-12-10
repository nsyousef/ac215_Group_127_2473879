"""Modal LLM service deployment wrapper module"""

import pulumi
from pulumi_command import local
from typing import Optional, Dict


class ModalLLMService(pulumi.ComponentResource):
    """Deploys Modal LLM service via modal CLI"""

    def __init__(
        self,
        name: str,
        model_size: str = "27b",
        gpu_type: str = "H200",
        max_containers: int = 1,
        scaledown_window: int = 1000,
        modal_token_id: str = None,
        modal_token_secret: str = None,
        modal_username: str = "nsyousef",
        tags: Optional[Dict] = None,
        opts: Optional[pulumi.ResourceOptions] = None,
    ):
        """
        Initialize the ModalLLMService component.

        Args:
            name: Resource name prefix
            model_size: Model size ('4b' or '27b')
            gpu_type: GPU type for Modal (e.g., 'H200', 'A10G')
            max_containers: Maximum number of containers for Modal
            scaledown_window: Scaledown window for Modal
            modal_token_id: Modal API token ID
            modal_token_secret: Modal API token secret
            modal_username: Modal username for URL construction
            tags: Resource tags/labels
            opts: Pulumi resource options
        """
        super().__init__("pibu:llm:ModalLLMService", name, None, opts)

        # Validate model size
        if model_size not in ["4b", "27b"]:
            raise ValueError(f"Invalid model_size: {model_size}. Must be '4b' or '27b'")

        model_name = f"medgemma-{model_size}"
        max_tokens = "500" if model_size == "4b" else "700"
        # Use parameter values instead of hardcoded strings
        max_containers_str = str(max_containers)
        scaledown_window_str = str(scaledown_window)
        # Construct deployment command
        deploy_cmd = f"""
cd ../llm && \
export MODAL_MODEL_NAME="{model_name}" && \
export MODAL_MAX_TOKENS="{max_tokens}" && \
export MODAL_GPU="{gpu_type}" && \
export MODAL_MAX_CONTAINERS="{max_containers_str}" && \
export MODAL_SCALEDOWN_WINDOW="{scaledown_window_str}" && \
modal deploy llm_modal.py
"""

        # Environment variables for Modal deployment
        env_vars = {
            "MODAL_MODEL_NAME": model_name,
            "MODAL_MAX_TOKENS": max_tokens,
            "MODAL_GPU": gpu_type,
            "MODAL_MAX_CONTAINERS": max_containers_str,
            "MODAL_SCALEDOWN_WINDOW": scaledown_window_str,
        }

        # Add Modal credentials if provided
        if modal_token_id and modal_token_secret:
            env_vars["MODAL_TOKEN_ID"] = modal_token_id
            env_vars["MODAL_TOKEN_SECRET"] = modal_token_secret

        # Deploy via modal CLI
        self.deployment = local.Command(
            f"{name}-deploy",
            create=deploy_cmd,
            update=deploy_cmd,
            environment=env_vars,
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Construct Modal endpoint URLs (deterministic based on Modal app name)
        # Modal app name from llm_modal.py: f"dermatology-llm-{MODEL_NAME.split('-')[1]}"
        # which becomes: dermatology-llm-4b or dermatology-llm-27b
        app_suffix = model_size  # "4b" or "27b"
        base_url = f"https://{modal_username}--dermatology-llm-{app_suffix}-dermatologyllm"

        self.explain_url = f"{base_url}-explain.modal.run"
        self.followup_url = f"{base_url}-ask-followup.modal.run"

        self.register_outputs(
            {
                "deployment": self.deployment,
                "explain_url": self.explain_url,
                "followup_url": self.followup_url,
                "model_size": model_size,
                "gpu_type": gpu_type,
            }
        )
