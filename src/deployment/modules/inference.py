"""Cloud Run inference service deployment module."""

import pulumi
import pulumi_gcp as gcp
import pulumi_docker as docker
from typing import Optional, Dict


class InferenceService(pulumi.ComponentResource):
    """Deploys the inference FastAPI service to Cloud Run"""

    def __init__(
        self,
        name: str,
        project_id: str,
        region: str,
        environment: str,
        memory: str = "4Gi",
        cpu: str = "2",
        min_instances: int = 0,
        max_instances: int = 10,
        tags: Optional[Dict] = None,
        opts: Optional[pulumi.ResourceOptions] = None,
    ):
        """
        Initialize the InferenceService component.

        Args:
            name: Resource name prefix
            project_id: GCP project ID
            region: GCP region
            environment: Environment (dev, prod)
            memory: Memory limit for Cloud Run
            cpu: CPU limit for Cloud Run
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            tags: Resource tags/labels
            opts: Pulumi resource options
        """
        super().__init__("pibu:inference:InferenceService", name, None, opts)

        # Service account for Cloud Run
        self.service_account = gcp.serviceaccount.Account(
            f"{name}-sa",
            account_id=f"inference-{environment}",
            display_name=f"Inference Service ({environment})",
            project=project_id,
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Build and push Docker image to GCR
        image_name = f"gcr.io/{project_id}/inference-cloud:{environment}"

        # Use absolute path mounted in container
        # inference-cloud is mounted at /inference-cloud in the deployment container
        self.image = docker.Image(
            f"{name}-image",
            build=docker.DockerBuildArgs(
                context="/inference-cloud",
                dockerfile="/inference-cloud/Dockerfile",
                platform="linux/amd64",  # Cloud Run requirement
            ),
            image_name=image_name,
            registry=docker.RegistryArgs(
                server="gcr.io",
            ),
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Deploy to Cloud Run
        self.service = gcp.cloudrunv2.Service(
            f"{name}-service",
            name=f"inference-{environment}",
            project=project_id,
            location=region,
            ingress="INGRESS_TRAFFIC_ALL",
            template=gcp.cloudrunv2.ServiceTemplateArgs(
                service_account=self.service_account.email,
                scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                    min_instance_count=min_instances, max_instance_count=max_instances
                ),
                containers=[
                    gcp.cloudrunv2.ServiceTemplateContainerArgs(
                        image=self.image.image_name,
                        ports=[gcp.cloudrunv2.ServiceTemplateContainerPortArgs(name="http1", container_port=8080)],
                        resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                            limits={"memory": memory, "cpu": cpu},
                            cpu_idle=True,  # Scale to zero
                            startup_cpu_boost=True,  # Fast cold starts
                        ),
                        envs=[
                            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="PORT", value="8080"),
                            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                                name="MODEL_CHECKPOINT_PATH", value="/app/models/test_best.pth"
                            ),
                            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="DEVICE", value="cpu"),
                            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="ENVIRONMENT", value=environment),
                        ],
                        # Liveness probe
                        liveness_probe=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeArgs(
                            http_get=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeHttpGetArgs(path="/health"),
                            initial_delay_seconds=10,
                            period_seconds=30,
                        ),
                        # Startup probe (for slow model loading)
                        startup_probe=gcp.cloudrunv2.ServiceTemplateContainerStartupProbeArgs(
                            http_get=gcp.cloudrunv2.ServiceTemplateContainerStartupProbeHttpGetArgs(path="/health"),
                            initial_delay_seconds=0,
                            period_seconds=5,
                            failure_threshold=12,  # 60 seconds total
                        ),
                    )
                ],
                timeout="300s",  # 5 min request timeout
            ),
            labels=tags,
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Allow public access
        gcp.cloudrunv2.ServiceIamMember(
            f"{name}-public-access",
            project=project_id,
            location=region,
            name=self.service.name,
            role="roles/run.invoker",
            member="allUsers",
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Extract service URL
        self.url = self.service.uri

        self.register_outputs(
            {
                "service": self.service,
                "url": self.url,
                "service_account": self.service_account,
                "image_name": self.image.image_name,
            }
        )
