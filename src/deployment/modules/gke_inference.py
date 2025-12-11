"""GKE inference service deployment module"""

import os
import yaml
import pulumi
import pulumi_gcp as gcp
import pulumi_docker as docker
import pulumi_kubernetes as k8s
from typing import Optional, Dict


class GKEInferenceService(pulumi.ComponentResource):
    """Deploys the inference FastAPI service to GKE"""

    def __init__(
        self,
        name: str,
        project_id: str,
        region: str,
        environment: str,
        cluster_name: pulumi.Output,
        cluster_endpoint: pulumi.Output,
        cluster_ca_certificate: pulumi.Output,
        memory: str = "4Gi",
        cpu: str = "2",
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_cpu_utilization: int = 70,
        target_memory_utilization: int = 80,
        model_gcs_path: str = "gs://apcomp215-datasets/test_best.pth",
        model_checkpoint_path: str = "/tmp/models/test_best.pth",
        device: str = "cpu",
        port: str = "8080",
        tags: Optional[Dict] = None,
        opts: Optional[pulumi.ResourceOptions] = None,
    ):
        """
        Initialize the GKEInferenceService component.

        Args:
            name: Resource name prefix
            project_id: GCP project ID
            region: GCP region
            environment: Environment (dev, prod)
            cluster_name: GKE cluster name
            cluster_endpoint: GKE cluster endpoint
            cluster_ca_certificate: GKE cluster CA certificate
            memory: Memory limit for containers
            cpu: CPU limit for containers
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            target_cpu_utilization: Target CPU utilization for HPA
            target_memory_utilization: Target memory utilization for HPA
            model_gcs_path: GCS path to model checkpoint (default: gs://apcomp215-datasets/test_best.pth)
            model_checkpoint_path: Local path where model will be downloaded (default: /tmp/models/test_best.pth)
            device: Device to run inference on (default: cpu)
            port: Port for the inference service (default: 8080)
            tags: Resource tags/labels
            opts: Pulumi resource options
        """
        super().__init__("pibu:gke:GKEInferenceService", name, None, opts)

        # Service account for Cloud resources (GCS access)
        self.service_account = gcp.serviceaccount.Account(
            f"{name}-sa",
            account_id=f"inference-{environment}",
            display_name=f"Inference Service ({environment})",
            project=project_id,
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Grant service account access to GCS bucket for model files
        gcp.storage.BucketIAMMember(
            f"{name}-gcs-access",
            bucket="apcomp215-datasets",
            role="roles/storage.objectViewer",
            member=self.service_account.email.apply(lambda email: f"serviceAccount:{email}"),
            opts=pulumi.ResourceOptions(parent=self, depends_on=[self.service_account]),
        )

        # Enable Artifact Registry API
        artifact_registry_api = gcp.projects.Service(
            f"{name}-artifact-registry-api",
            project=project_id,
            service="artifactregistry.googleapis.com",
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Create Artifact Registry repository if it doesn't exist
        repository = gcp.artifactregistry.Repository(
            f"{name}-repository",
            repository_id="pibu-ai-images",
            location=region,
            project=project_id,
            format="DOCKER",
            description="Docker images for Pibu.AI inference service",
            opts=pulumi.ResourceOptions(
                parent=self,
                depends_on=[artifact_registry_api],
            ),
        )

        # Build and push Docker image to Artifact Registry
        image_name = f"{region}-docker.pkg.dev/{project_id}/pibu-ai-images/inference-cloud:{environment}"

        # Docker build paths - configurable via environment variable or use default
        # These paths match the volume mounts in docker-shell.sh
        docker_context = os.getenv("DOCKER_BUILD_CONTEXT", "/ac215_Group_127_2473879/src")
        dockerfile_path = os.path.join(docker_context, "inference-cloud/Dockerfile")

        # Get GCP access token for Artifact Registry authentication
        client_config = gcp.organizations.get_client_config()

        # Build Docker image with explicit registry authentication
        self.image = docker.Image(
            f"{name}-image",
            build=docker.DockerBuildArgs(
                context=docker_context,
                dockerfile=dockerfile_path,
                platform="linux/amd64",
            ),
            image_name=image_name,
            registry=docker.RegistryArgs(
                server=f"{region}-docker.pkg.dev",
                username="oauth2accesstoken",
                password=client_config.access_token,
            ),
            opts=pulumi.ResourceOptions(
                parent=self,
                depends_on=[repository],
            ),
        )

        # Generate kubeconfig for GKE cluster
        k8s_info = pulumi.Output.all(cluster_name, cluster_endpoint, cluster_ca_certificate)

        def make_kubeconfig(info):
            cluster_name_val, endpoint, ca_cert = info
            context_name = f"{project_id}_{region}_{cluster_name_val}"

            kubeconfig = {
                "apiVersion": "v1",
                "kind": "Config",
                "clusters": [
                    {
                        "name": context_name,
                        "cluster": {
                            "certificate-authority-data": ca_cert,
                            "server": f"https://{endpoint}",
                        },
                    }
                ],
                "contexts": [
                    {
                        "name": context_name,
                        "context": {"cluster": context_name, "user": context_name},
                    }
                ],
                "current-context": context_name,
                "users": [
                    {
                        "name": context_name,
                        "user": {
                            "exec": {
                                "apiVersion": "client.authentication.k8s.io/v1beta1",
                                "command": "gke-gcloud-auth-plugin",
                                "installHint": "Install gke-gcloud-auth-plugin for use with kubectl",
                                "provideClusterInfo": True,
                                "interactiveMode": "Never",
                            }
                        },
                    }
                ],
            }
            return yaml.dump(kubeconfig, default_flow_style=False)

        cluster_kubeconfig = k8s_info.apply(make_kubeconfig)

        # Create Kubernetes provider for the GKE cluster
        k8s_provider = k8s.Provider(
            f"{name}-k8s-provider",
            kubeconfig=cluster_kubeconfig,
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Create Kubernetes namespace for application deployments
        namespace_resource = k8s.core.v1.Namespace(
            f"{name}-namespace",
            metadata=k8s.meta.v1.ObjectMetaArgs(
                name=f"inference-{environment}",
                labels=tags,
            ),
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider),
        )
        namespace = namespace_resource.metadata.name

        # Create Kubernetes service account
        k8s_service_account = k8s.core.v1.ServiceAccount(
            f"{name}-k8s-sa",
            metadata=k8s.meta.v1.ObjectMetaArgs(
                name="inference-sa",
                namespace=namespace,
                annotations={
                    "iam.gke.io/gcp-service-account": self.service_account.email,
                },
            ),
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider),
        )

        # Bind GCP service account to Kubernetes service account (Workload Identity)
        # Build the Workload Identity member string: serviceAccount:PROJECT_ID.svc.id.goog[namespace/ksa_name]
        wi_member = namespace.apply(lambda ns: f"serviceAccount:{project_id}.svc.id.goog[{ns}/inference-sa]")

        # Construct the full GSA resource ID
        gsa_full_id = self.service_account.email.apply(lambda email: f"projects/{project_id}/serviceAccounts/{email}")

        # Grant the KSA permission to act as the GSA using IAMMember (single binding)
        gcp.serviceaccount.IAMMember(
            f"{name}-workload-identity-binding",
            service_account_id=gsa_full_id,
            role="roles/iam.workloadIdentityUser",
            member=wi_member,
            opts=pulumi.ResourceOptions(parent=self),
        )

        # ConfigMap for environment variables
        config_map = k8s.core.v1.ConfigMap(
            f"{name}-config",
            metadata=k8s.meta.v1.ObjectMetaArgs(
                name=f"inference-{environment}-config",
                namespace=namespace,
            ),
            data={
                "MODEL_CHECKPOINT_PATH": model_checkpoint_path,
                "MODEL_GCS_PATH": model_gcs_path,
                "DEVICE": device,
                "ENVIRONMENT": environment,
                "PORT": port,
                "GCP_PROJECT": project_id,
            },
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider),
        )

        # Deployment
        app_labels = {"app": f"inference-{environment}"}
        deployment = k8s.apps.v1.Deployment(
            f"{name}-deployment",
            metadata=k8s.meta.v1.ObjectMetaArgs(
                name=f"inference-{environment}",
                namespace=namespace,
                labels=tags,
            ),
            spec=k8s.apps.v1.DeploymentSpecArgs(
                replicas=min_replicas,
                progress_deadline_seconds=1200,  # allow up to 20 minutes for initial model load/boot
                selector=k8s.meta.v1.LabelSelectorArgs(match_labels=app_labels),
                template=k8s.core.v1.PodTemplateSpecArgs(
                    metadata=k8s.meta.v1.ObjectMetaArgs(labels=app_labels),
                    spec=k8s.core.v1.PodSpecArgs(
                        service_account_name="inference-sa",
                        containers=[
                            k8s.core.v1.ContainerArgs(
                                name="inference",
                                image=self.image.image_name,
                                ports=[k8s.core.v1.ContainerPortArgs(container_port=8080, name="http")],
                                env_from=[
                                    k8s.core.v1.EnvFromSourceArgs(
                                        config_map_ref=k8s.core.v1.ConfigMapEnvSourceArgs(name=config_map.metadata.name)
                                    )
                                ],
                                resources=k8s.core.v1.ResourceRequirementsArgs(
                                    requests={"memory": "3Gi", "cpu": "2"},
                                    limits={"memory": memory, "cpu": cpu},
                                ),
                                # Readiness probe
                                readiness_probe=k8s.core.v1.ProbeArgs(
                                    http_get=k8s.core.v1.HTTPGetActionArgs(path="/health", port=8080),
                                    initial_delay_seconds=10,
                                    period_seconds=10,
                                ),
                                # Liveness probe
                                liveness_probe=k8s.core.v1.ProbeArgs(
                                    http_get=k8s.core.v1.HTTPGetActionArgs(path="/health", port=8080),
                                    initial_delay_seconds=10,
                                    period_seconds=30,
                                ),
                                # Startup probe (for model download)
                                startup_probe=k8s.core.v1.ProbeArgs(
                                    http_get=k8s.core.v1.HTTPGetActionArgs(path="/health", port=8080),
                                    initial_delay_seconds=0,
                                    period_seconds=5,
                                    timeout_seconds=30,
                                    failure_threshold=60,  # ample time for large model download/startup
                                ),
                                volume_mounts=[
                                    k8s.core.v1.VolumeMountArgs(
                                        name="model-cache",
                                        mount_path="/tmp/models",
                                    )
                                ],
                            )
                        ],
                        volumes=[
                            k8s.core.v1.VolumeArgs(
                                name="model-cache",
                                empty_dir=k8s.core.v1.EmptyDirVolumeSourceArgs(),
                            )
                        ],
                    ),
                ),
            ),
            opts=pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=[k8s_service_account, config_map],
            ),
        )

        # Service (LoadBalancer)
        service = k8s.core.v1.Service(
            f"{name}-service",
            metadata=k8s.meta.v1.ObjectMetaArgs(
                name=f"inference-{environment}",
                namespace=namespace,
                labels=tags,
            ),
            spec=k8s.core.v1.ServiceSpecArgs(
                type="LoadBalancer",
                selector=app_labels,
                ports=[
                    k8s.core.v1.ServicePortArgs(
                        port=80,
                        target_port=8080,
                        protocol="TCP",
                        name="http",
                    )
                ],
            ),
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider),
        )

        # Horizontal Pod Autoscaler
        hpa = k8s.autoscaling.v2.HorizontalPodAutoscaler(
            f"{name}-hpa",
            metadata=k8s.meta.v1.ObjectMetaArgs(
                name=f"inference-{environment}",
                namespace=namespace,
            ),
            spec=k8s.autoscaling.v2.HorizontalPodAutoscalerSpecArgs(
                scale_target_ref=k8s.autoscaling.v2.CrossVersionObjectReferenceArgs(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deployment.metadata.name,
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=[
                    k8s.autoscaling.v2.MetricSpecArgs(
                        type="Resource",
                        resource=k8s.autoscaling.v2.ResourceMetricSourceArgs(
                            name="cpu",
                            target=k8s.autoscaling.v2.MetricTargetArgs(
                                type="Utilization",
                                average_utilization=target_cpu_utilization,
                            ),
                        ),
                    ),
                    k8s.autoscaling.v2.MetricSpecArgs(
                        type="Resource",
                        resource=k8s.autoscaling.v2.ResourceMetricSourceArgs(
                            name="memory",
                            target=k8s.autoscaling.v2.MetricTargetArgs(
                                type="Utilization",
                                average_utilization=target_memory_utilization,
                            ),
                        ),
                    ),
                ],
            ),
            opts=pulumi.ResourceOptions(parent=self, provider=k8s_provider, depends_on=[deployment]),
        )

        # Extract service URL (LoadBalancer IP)
        self.url = service.status.apply(
            lambda status: (
                f"http://{status.load_balancer.ingress[0].ip}"
                if status and status.load_balancer and status.load_balancer.ingress
                else "pending"
            )
        )

        self.register_outputs(
            {
                "service_account": self.service_account,
                "image_name": self.image.image_name,
                "deployment": deployment,
                "service": service,
                "hpa": hpa,
                "url": self.url,
            }
        )
