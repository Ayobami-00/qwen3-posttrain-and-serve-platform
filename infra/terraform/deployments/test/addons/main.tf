terraform {
  required_version = "= 1.16.6"
  required_providers {
    helm = {
      source = "hashicorp/helm", version = "3.3.0"
    }
    kubernetes = {
      source = "hashicorp/kubernetes", version = "3.2.1"
    }
  }
}
variable "stack" {
  type = any
}
variable "region" {
  type = string
}
variable "logs_bucket" {
  type = string
}
provider "helm" {
  kubernetes = {
    host                   = var.stack.cluster_endpoint
    cluster_ca_certificate = base64decode(var.stack.cluster_ca)
    exec = {
      api_version = "client.authentication.k8s.io/v1beta1", command = "aws", args = ["eks", "get-token", "--cluster-name", var.stack.cluster_name, "--region", var.region]
    }
  }
}
provider "kubernetes" {
  host                   = var.stack.cluster_endpoint
  cluster_ca_certificate = base64decode(var.stack.cluster_ca)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.stack.cluster_name, "--region", var.region]
  }
}
resource "kubernetes_storage_class_v1" "gp3" {
  metadata {
    name = "gp3"
  }
  storage_provisioner = "ebs.csi.aws.com"
  reclaim_policy      = "Delete"
  volume_binding_mode = "WaitForFirstConsumer"
  parameters = {
    type               = "gp3", encrypted = "true"
    tagSpecification_1 = "Project=qps"
    tagSpecification_2 = "Environment=test"
  }
}
resource "helm_release" "alb" {
  name       = "aws-load-balancer-controller"
  namespace  = "kube-system"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  version    = "3.6.0"
  atomic     = true
  timeout    = 600
  values = [yamlencode({
    clusterName  = var.stack.cluster_name, region = var.region, vpcId = var.stack.vpc_id
    replicaCount = 1
    serviceAccount = {
      create = true, name = "aws-load-balancer-controller"
    }
    resources = {
      requests = {
        cpu = "100m", memory = "128Mi"
        }, limits = {
        memory = "384Mi"
      }
    }
  })]
}
resource "helm_release" "monitoring" {
  name             = "qps-monitoring"
  namespace        = "monitoring"
  create_namespace = true
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = "92.9.0"
  atomic           = true
  timeout          = 900
  values = [templatefile("${path.module}/monitoring.yaml.tftpl", {
    public_url = var.stack.url
  })]
  depends_on = [kubernetes_storage_class_v1.gp3]
}
resource "helm_release" "loki" {
  name             = "qps-loki"
  namespace        = "monitoring"
  create_namespace = true
  repository       = "https://grafana.github.io/helm-charts"
  chart            = "loki"
  version          = "7.3.0"
  atomic           = true
  timeout          = 600
  values = [templatefile("${path.module}/loki.yaml.tftpl", {
    region = var.region, bucket = var.logs_bucket
  })]
  depends_on = [kubernetes_storage_class_v1.gp3]
}
resource "helm_release" "collector" {
  name       = "qps-otel"
  namespace  = "monitoring"
  repository = "https://open-telemetry.github.io/opentelemetry-helm-charts"
  chart      = "opentelemetry-collector"
  version    = "0.176.0"
  atomic     = true
  timeout    = 600
  values     = [file("${path.module}/otel.yaml")]
  depends_on = [helm_release.loki]
}
