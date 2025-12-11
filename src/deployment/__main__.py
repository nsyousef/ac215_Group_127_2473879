"""
Pibu.AI Infrastructure Deployment
Deploys GKE inference service and manages Modal LLM deployment
"""

import pulumi
from create_network import create_network
from modules.gke_cluster import GKECluster
from modules.gke_inference import GKEInferenceService
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
# NETWORK
# ============================================================================
app_name = f"pibu-ai-{environment}"
network, subnet, router, nat = create_network(region, app_name)

# ============================================================================
# GKE CLUSTER
# ============================================================================
gke_cluster = GKECluster(
    "gke",
    project_id=project_id,
    region=region,
    environment=environment,
    network=network,
    subnet=subnet,
    node_count=config.get_int("gke_node_count") or 1,
    min_node_count=config.get_int("gke_min_node_count") or 1,
    max_node_count=config.get_int("gke_max_node_count") or 3,
    machine_type=config.get("gke_machine_type") or "e2-standard-4",
    disk_size_gb=config.get_int("gke_disk_size_gb") or 30,
    tags=tags,
)

# ============================================================================
# GKE INFERENCE SERVICE
# ============================================================================
inference_service = GKEInferenceService(
    "inference",
    project_id=project_id,
    region=region,
    environment=environment,
    cluster_name=gke_cluster.cluster_name,
    cluster_endpoint=gke_cluster.cluster_endpoint,
    cluster_ca_certificate=gke_cluster.cluster_ca_certificate,
    memory=config.get("inference_memory") or "4Gi",
    cpu=config.get("inference_cpu") or "2",
    min_replicas=config.get_int("inference_min_replicas") or (2 if environment == "prod" else 1),
    max_replicas=config.get_int("inference_max_replicas") or 10,
    target_cpu_utilization=config.get_int("inference_target_cpu") or 70,
    target_memory_utilization=config.get_int("inference_target_memory") or 80,
    model_gcs_path=config.get("inference_model_gcs_path") or "gs://apcomp215-datasets/test_best.pth",
    model_checkpoint_path=config.get("inference_model_checkpoint_path") or "/tmp/models/test_best.pth",
    device=config.get("inference_device") or "cpu",
    port=config.get("inference_port") or "8080",
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
    modal_username=config.get("modal_username") or "nsyousef",
    tags=tags,
)

# ============================================================================
# STACK OUTPUTS
# ============================================================================
pulumi.export("environment", environment)
pulumi.export("region", region)
pulumi.export("project_id", project_id)

# GKE cluster outputs
pulumi.export("gke_cluster_name", gke_cluster.cluster_name)
pulumi.export("gke_cluster_endpoint", gke_cluster.cluster_endpoint)

# Inference service outputs
pulumi.export("inference_url", inference_service.url)
pulumi.export("inference_image", inference_service.image.image_name)

# LLM service outputs
pulumi.export("llm_explain_url", modal_llm.explain_url)
pulumi.export("llm_followup_url", modal_llm.followup_url)
pulumi.export("llm_time_tracking_url", modal_llm.time_tracking_url)
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
        llm_time_tracking_url=modal_llm.time_tracking_url,
    ).apply(
        lambda args: {
            "BASE_URL": args["inference_url"],
            "LLM_EXPLAIN_URL": args["llm_explain_url"],
            "LLM_FOLLOWUP_URL": args["llm_followup_url"],
            "LLM_TIME_TRACKING_URL": args["llm_time_tracking_url"],
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
