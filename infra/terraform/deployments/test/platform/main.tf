terraform {
  required_version = "= 1.16.6"
  required_providers {
    helm = {
      source = "hashicorp/helm", version = "3.3.0"
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
  chart            = "${path.module}/../../../../../deploy/helm/qps"
  atomic           = true
  timeout          = 900
  values = [yamlencode({
    images = {
      api = var.images.api, web = var.images.web, operator = var.images.operator
    }
    apiSecretArn      = var.secrets.api
    operatorSecretArn = var.secrets.operator
    region            = var.region
    targetGroups      = var.stack.target_groups
  })]
}
