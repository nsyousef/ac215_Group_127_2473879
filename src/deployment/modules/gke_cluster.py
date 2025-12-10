"""GKE cluster deployment module"""

import pulumi
import pulumi_gcp as gcp
from typing import Optional, Dict


class GKECluster(pulumi.ComponentResource):
    """Deploys a GKE cluster with Workload Identity enabled"""

    def __init__(
        self,
        name: str,
        project_id: str,
        region: str,
        environment: str,
        network: gcp.compute.Network,
        subnet: gcp.compute.Subnetwork,
        node_count: int = 1,
        min_node_count: int = 1,
        max_node_count: int = 3,
        machine_type: str = "e2-standard-4",
        disk_size_gb: int = 30,
        tags: Optional[Dict] = None,
        opts: Optional[pulumi.ResourceOptions] = None,
    ):
        """
        Initialize the GKECluster component.

        Args:
            name: Resource name prefix
            project_id: GCP project ID
            region: GCP region
            environment: Environment (dev, prod)
            network: Custom VPC network
            subnet: Custom subnet
            node_count: Initial node count
            min_node_count: Minimum node count for autoscaling
            max_node_count: Maximum node count for autoscaling
            machine_type: GCP machine type for nodes
            disk_size_gb: Disk size in GB for node VMs (default: 30)
            tags: Resource tags/labels
            opts: Pulumi resource options
        """
        super().__init__("pibu:gke:GKECluster", name, None, opts)

        # Enable required APIs
        container_api = gcp.projects.Service(
            f"{name}-container-api",
            project=project_id,
            service="container.googleapis.com",
            opts=pulumi.ResourceOptions(parent=self),
        )

        # Build dependencies list - network and subnet are required
        depends_on = [container_api, network, subnet]

        # Create GKE cluster
        self.cluster = gcp.container.Cluster(
            f"{name}-cluster",
            name=f"pibu-ai-{environment}",
            project=project_id,
            location=region,
            # Use regional cluster for high availability
            # Remove specific zones to use all zones in region
            initial_node_count=1,
            remove_default_node_pool=True,
            deletion_protection=False,  # Allow cluster deletion for dev environments
            # Network configuration - use custom network/subnet
            network=network.name,
            subnetwork=subnet.name,
            # Enable Workload Identity
            workload_identity_config=gcp.container.ClusterWorkloadIdentityConfigArgs(
                workload_pool=f"{project_id}.svc.id.goog",
            ),
            # Private cluster configuration
            private_cluster_config=gcp.container.ClusterPrivateClusterConfigArgs(
                enable_private_nodes=True,  # Nodes use private IPs only
                enable_private_endpoint=False,  # Control plane accessible via public endpoint
                master_ipv4_cidr_block="172.0.0.0/28",  # CIDR for GKE control plane
            ),
            # Enable basic features
            logging_service="logging.googleapis.com/kubernetes",
            monitoring_service="monitoring.googleapis.com/kubernetes",
            # Release channel for automatic updates
            release_channel=gcp.container.ClusterReleaseChannelArgs(
                channel="REGULAR",
            ),
            opts=pulumi.ResourceOptions(
                parent=self,
                depends_on=depends_on,
            ),
        )

        # Create node pool
        self.node_pool = gcp.container.NodePool(
            f"{name}-node-pool",
            name=f"pibu-ai-{environment}-pool",
            project=project_id,
            location=region,
            cluster=self.cluster.name,
            initial_node_count=node_count,
            # Node configuration
            node_config=gcp.container.NodePoolNodeConfigArgs(
                machine_type=machine_type,
                image_type="cos_containerd",  # Container-Optimized OS with containerd runtime
                disk_size_gb=disk_size_gb,
                disk_type="pd-standard",
                # Enable Workload Identity on nodes
                workload_metadata_config=gcp.container.NodePoolNodeConfigWorkloadMetadataConfigArgs(
                    mode="GKE_METADATA",
                ),
                # OAuth scopes for node service account permissions
                oauth_scopes=[
                    "https://www.googleapis.com/auth/devstorage.read_only",  # Read from GCS
                    "https://www.googleapis.com/auth/logging.write",  # Write logs to Cloud Logging
                    "https://www.googleapis.com/auth/monitoring",  # Send metrics to Cloud Monitoring
                    "https://www.googleapis.com/auth/servicecontrol",  # Service control access
                    "https://www.googleapis.com/auth/service.management.readonly",  # Read service management
                    "https://www.googleapis.com/auth/trace.append",  # Write traces to Cloud Trace
                ],
                labels=tags,
            ),
            # Autoscaling configuration
            autoscaling=gcp.container.NodePoolAutoscalingArgs(
                min_node_count=min_node_count,
                max_node_count=max_node_count,
            ),
            # Management configuration
            management=gcp.container.NodePoolManagementArgs(
                auto_repair=True,
                auto_upgrade=True,
            ),
            opts=pulumi.ResourceOptions(
                parent=self,
                # Ignore server-managed fields to avoid unnecessary drift updates.
                # When you need to intentionally change any of these (e.g., machine type,
                # autoscaling bounds), temporarily remove the relevant entry before pulumi up.
                ignore_changes=[
                    "node_config",
                    "node_config.workload_metadata_config",
                    "node_config.oauth_scopes",
                    "node_config.labels",
                    "management",
                    "autoscaling",
                ],
            ),
        )

        # Export cluster details
        self.cluster_name = self.cluster.name
        self.cluster_endpoint = self.cluster.endpoint
        self.cluster_ca_certificate = self.cluster.master_auth.cluster_ca_certificate

        self.register_outputs(
            {
                "cluster": self.cluster,
                "node_pool": self.node_pool,
                "cluster_name": self.cluster_name,
                "cluster_endpoint": self.cluster_endpoint,
            }
        )
