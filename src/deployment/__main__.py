"""
Pibu.AI Infrastructure Deployment
Deploys Cloud Run inference service and manages Modal LLM deployment
"""

import pulumi
from modules.inference import InferenceService
from modules.modal_llm import ModalLLMService

# Get configuration
config = pulumi.Config()
gcp_config = pulumi.Config("gcp")
project_id = gcp_config.require("project")
region = gcp_config.get("region") or "us-east1"
environment = pulumi.get_stack()  # dev, staging, prod

# Tags for resource organization
tags = {"project": "pibu-ai", "environment": environment, "managed-by": "pulumi"}

# ============================================================================
# CLOUD RUN INFERENCE SERVICE
# ============================================================================
inference_service = InferenceService(
    "inference",
    project_id=project_id,
    region=region,
    environment=environment,
    memory=config.get("inference_memory") or "4Gi",
    cpu=config.get("inference_cpu") or "2",
    min_instances=config.get_int("inference_min_instances") or (1 if environment == "prod" else 0),
    max_instances=config.get_int("inference_max_instances") or 10,
    tags=tags,
)

# ============================================================================
# MODAL LLM SERVICE
# ============================================================================
modal_llm = ModalLLMService(
    "llm",
    model_size=config.get("llm_model_size") or "27b",
    gpu_type=config.get("llm_gpu") or "H200",
    modal_token_id=config.get_secret("modal_token_id"),
    modal_token_secret=config.get_secret("modal_token_secret"),
    modal_username=config.get("modal_username") or "tanushkmr2001",
    tags=tags,
)

# ============================================================================
# STACK OUTPUTS
# ============================================================================
pulumi.export("environment", environment)
pulumi.export("region", region)
pulumi.export("project_id", project_id)

# Inference service outputs
pulumi.export("inference_url", inference_service.url)
pulumi.export("inference_service_name", inference_service.service.name)
pulumi.export("inference_image", inference_service.image.image_name)

# LLM service outputs
pulumi.export("llm_explain_url", modal_llm.explain_url)
pulumi.export("llm_followup_url", modal_llm.followup_url)
pulumi.export(
    "llm_model_size", modal_llm.deployment.environment.apply(lambda env: env.get("MODAL_MODEL_NAME", "unknown"))
)

# Frontend configuration (for building Electron app)
pulumi.export(
    "frontend_config",
    pulumi.Output.all(
        inference_url=inference_service.url,
        llm_explain_url=modal_llm.explain_url,
        llm_followup_url=modal_llm.followup_url,
    ).apply(
        lambda args: {
            "BASE_URL": args["inference_url"],
            "LLM_EXPLAIN_URL": args["llm_explain_url"],
            "LLM_FOLLOWUP_URL": args["llm_followup_url"],
        }
    ),
)

# Summary output
pulumi.export(
    "deployment_summary",
    pulumi.Output.all(inference_url=inference_service.url, llm_explain_url=modal_llm.explain_url).apply(
        lambda args: f"""
Deployment Complete!

Inference API: {args["inference_url"]}
LLM Explain:   {args["llm_explain_url"]}

Test the services:
  curl {args["inference_url"]}/health
"""
    ),
)
