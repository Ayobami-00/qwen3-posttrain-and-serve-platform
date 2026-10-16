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
variable "images" {
  type = map(string)
}
variable "secrets" {
  type = map(string)
}
variable "environment_deadline" {
  type = number
}
locals {
  platform_chart = "${path.module}/../../../../../deploy/helm/qps"
  chart_content_sha = sha256(jsonencode({
    for name in fileset(local.platform_chart, "**") : name => filesha256("${local.platform_chart}/${name}")
  }))
  crds = {
    for filename in fileset("${local.platform_chart}/crds", "*.yaml") :
    filename => jsondecode(file("${local.platform_chart}/crds/${filename}"))
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
data "kubernetes_resources" "existing_crds" {
  api_version = "apiextensions.k8s.io/v1"
  kind        = "CustomResourceDefinition"
}
import {
  for_each = {
    for filename, manifest in local.crds : filename => manifest
    if contains([for object in data.kubernetes_resources.existing_crds.objects : object.metadata.name], manifest.metadata.name)
  }
  to = kubernetes_manifest.crd[each.key]
  id = "apiVersion=apiextensions.k8s.io/v1,kind=CustomResourceDefinition,name=${each.value.metadata.name}"
}
resource "kubernetes_manifest" "crd" {
  for_each = local.crds
  manifest = each.value
  field_manager {
    name            = "qps-terraform"
    force_conflicts = true
  }
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
resource "helm_release" "qps" {
  name             = "qps"
  namespace        = "qps"
  create_namespace = true
  chart            = local.platform_chart
  atomic           = true
  skip_crds        = true
  timeout          = 900
  depends_on       = [kubernetes_manifest.crd]
  values = [yamlencode({
    chartContentSha = local.chart_content_sha
    images = {
      api = var.images.api, web = var.images.web, operator = var.images.operator
    }
    apiSecretArn      = var.secrets.api
    operatorSecretArn = var.secrets.operator
    region            = var.region
    apiEnvironment    = { QPS_ENVIRONMENT_DEADLINE = tostring(var.environment_deadline) }
    targetGroups      = var.stack.target_groups
  })]
}
