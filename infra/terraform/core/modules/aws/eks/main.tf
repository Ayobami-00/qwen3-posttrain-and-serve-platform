terraform {
  required_providers {
    time = {
      source = "hashicorp/time", version = "0.14.2"
    }
  }
}
variable "prefix" {
  type = string
}
variable "account_id" {
  type = string
}
variable "region" {
  type = string
}
variable "subnet_ids" {
  type = list(string)
}
variable "provision_role_arn" {
  type = string
}
variable "artifact_bucket_arn" {
  type = string
}
variable "api_secret_arn" {
  type = string
}
variable "operator_secret_arn" {
  type = string
}
variable "logs_bucket_arn" {
  type = string
}
locals {
  pod_trust = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = {
        Service = "pods.eks.amazonaws.com"
      }, Action = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
  roles = {
    api = {
      namespace = "qps", service_account = "qps-api"
      }, operator = {
      namespace = "qps", service_account = "qps-operator"
      }, alb = {
      namespace = "kube-system", service_account = "aws-load-balancer-controller"
      }, ebs = {
      namespace = "kube-system", service_account = "ebs-csi-controller-sa"
      }, cni = {
      namespace = "kube-system", service_account = "aws-node"
      }, loki = {
      namespace = "monitoring", service_account = "qps-loki"
    }
  }
}
resource "aws_iam_role" "cluster" {
  name = "${var.prefix}-eks-cluster"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = {
        Service = "eks.amazonaws.com"
      }, Action = "sts:AssumeRole"
    }]
  })
}
resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}
resource "aws_cloudwatch_log_group" "cluster" {
  name              = "/aws/eks/${var.prefix}/cluster"
  retention_in_days = 1
}
resource "aws_eks_cluster" "main" {
  name                      = var.prefix
  role_arn                  = aws_iam_role.cluster.arn
  version                   = "1.36"
  enabled_cluster_log_types = ["api", "audit", "authenticator"]
  upgrade_policy {
    support_type = "STANDARD"
  }
  access_config {
    authentication_mode                         = "API"
    bootstrap_cluster_creator_admin_permissions = false
  }
  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = false
  }
  depends_on = [aws_iam_role_policy_attachment.cluster, aws_cloudwatch_log_group.cluster]
}
resource "aws_eks_access_entry" "provision" {
  cluster_name  = aws_eks_cluster.main.name
  principal_arn = var.provision_role_arn
  type          = "STANDARD"
}
resource "aws_eks_access_policy_association" "provision" {
  cluster_name  = aws_eks_cluster.main.name
  principal_arn = aws_eks_access_entry.provision.principal_arn
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  access_scope {
    type = "cluster"
  }
}
resource "aws_iam_role" "node" {
  name = "${var.prefix}-eks-node"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = {
        Service = "ec2.amazonaws.com"
      }, Action = "sts:AssumeRole"
    }]
  })
}
resource "aws_iam_role_policy_attachment" "node" {
  for_each   = toset(["AmazonEKSWorkerNodePolicy", "AmazonEC2ContainerRegistryPullOnly", "AmazonEKS_CNI_Policy"])
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/${each.key}"
}
resource "aws_iam_role" "pod" {
  for_each           = local.roles
  name               = "${var.prefix}-${each.key}"
  assume_role_policy = local.pod_trust
}
resource "aws_eks_pod_identity_association" "pod" {
  for_each        = local.roles
  cluster_name    = aws_eks_cluster.main.name
  namespace       = each.value.namespace
  service_account = each.value.service_account
  role_arn        = aws_iam_role.pod[each.key].arn
}
resource "aws_iam_role_policy_attachment" "cni" {
  role       = aws_iam_role.pod["cni"].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}
resource "aws_iam_role_policy_attachment" "ebs" {
  role       = aws_iam_role.pod["ebs"].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}
resource "aws_eks_addon" "base" {
  for_each                    = toset(["vpc-cni", "kube-proxy", "coredns", "eks-pod-identity-agent", "aws-ebs-csi-driver"])
  cluster_name                = aws_eks_cluster.main.name
  addon_name                  = each.key
  addon_version               = jsondecode(file("${path.module}/addons.json"))[each.key]
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  configuration_values = each.key == "vpc-cni" ? jsonencode({
    enableNetworkPolicy = "true"
  }) : null
  # Host-network bootstrap add-ons (especially Pod Identity) must be installed
  # while nodes are starting. CNI credential delivery cannot wait for NodeReady.
  depends_on = [aws_iam_role_policy_attachment.cni, aws_iam_role_policy_attachment.ebs]
}
resource "aws_launch_template" "cpu" {
  name_prefix = "${var.prefix}-cpu-"
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 30
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }
  tag_specifications {
    resource_type = "instance"
    tags          = { Project = "qps", Environment = "test", Name = "${var.prefix}-cpu" }
  }
}
resource "aws_iam_role_policy" "node_launch_validation" {
  name = "${var.prefix}-node-launch-validation"
  role = split("/", var.provision_role_arn)[1]
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect   = "Allow", Action = "ec2:RunInstances"
        Resource = "arn:aws:ec2:${var.region}:${var.account_id}:instance/*"
        Condition = {
          ArnEquals    = { "ec2:LaunchTemplate" = aws_launch_template.cpu.arn }
          StringEquals = { "ec2:InstanceType" = "m7i.large" }
        }
      },
      {
        Effect = "Allow", Action = "ec2:RunInstances"
        Resource = concat(
          [aws_launch_template.cpu.arn, "arn:aws:ec2:${var.region}::image/*"],
          [for kind in ["volume", "network-interface", "subnet", "security-group", "key-pair"] : "arn:aws:ec2:${var.region}:${var.account_id}:${kind}/*"]
        )
        Condition = { ArnEquals = { "ec2:LaunchTemplate" = aws_launch_template.cpu.arn } }
      }
    ]
  })
}
resource "time_sleep" "node_launch_permission" {
  create_duration = "30s"
  triggers = {
    policy = sha256(aws_iam_role_policy.node_launch_validation.policy)
  }
  depends_on = [aws_iam_role_policy.node_launch_validation]
}
resource "aws_eks_node_group" "cpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.prefix}-cpu"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids
  instance_types  = ["m7i.large"]
  ami_type        = "AL2023_x86_64_STANDARD"
  capacity_type   = "ON_DEMAND"
  launch_template {
    id      = aws_launch_template.cpu.id
    version = aws_launch_template.cpu.latest_version
  }
  scaling_config {
    desired_size = 2
    min_size     = 2
    max_size     = 2
  }
  update_config {
    max_unavailable = 1
  }
  labels = {
    "qps.dev/pool" = "platform"
  }
  depends_on = [aws_iam_role_policy_attachment.node, aws_iam_role_policy_attachment.cni, time_sleep.node_launch_permission]
}
resource "aws_iam_role_policy" "api" {
  role = aws_iam_role.pod["api"].id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect = "Allow", Action = ["s3:ListBucket", "s3:GetBucketLocation"], Resource = var.artifact_bucket_arn
      },
      {
        Effect    = "Allow", Action = ["s3:ListBucketVersions"], Resource = var.artifact_bucket_arn,
        Condition = { StringLike = { "s3:prefix" = ["test/artifacts/*", "test/staging/*", "test/allocations/*"] } }
      },
      {
        # This listing action does not support the s3:prefix condition key.
        # Aborting uploads and deleting versions remain object-prefix scoped.
        Effect = "Allow", Action = ["s3:ListBucketMultipartUploads"], Resource = var.artifact_bucket_arn
      },
      {
        Effect = "Allow", Action = ["s3:DeleteObjectVersion"], Resource = ["${var.artifact_bucket_arn}/test/artifacts/*", "${var.artifact_bucket_arn}/test/staging/*", "${var.artifact_bucket_arn}/test/allocations/*"]
      },
      {
        Effect = "Allow", Action = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:DeleteObject", "s3:PutObjectTagging", "s3:GetObjectTagging", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"], Resource = "${var.artifact_bucket_arn}/test/*"
      },
      {
        Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = var.api_secret_arn
      }
    ]
  })
}
resource "aws_iam_role_policy" "operator" {
  role = aws_iam_role.pod["operator"].id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = var.operator_secret_arn
      },
      {
        Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*"
      },
      {
        Effect = "Allow", Action = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], Resource = "arn:aws:ecr:${var.region}:${var.account_id}:repository/${var.prefix}/worker"
      }
    ]
  })
}
resource "aws_iam_role_policy" "loki" {
  role = aws_iam_role.pod["loki"].id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect = "Allow", Action = ["s3:ListBucket", "s3:GetBucketLocation"], Resource = var.logs_bucket_arn
      },
      {
        Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], Resource = "${var.logs_bucket_arn}/*"
      }
    ]
  })
}
resource "aws_iam_role_policy" "alb" {
  role   = aws_iam_role.pod["alb"].id
  policy = file("${path.module}/alb-policy.json")
}
output "name" {
  value = aws_eks_cluster.main.name
}
output "endpoint" {
  value = aws_eks_cluster.main.endpoint
}
output "ca" {
  value = aws_eks_cluster.main.certificate_authority[0].data
}
output "security_group_id" {
  value = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}
