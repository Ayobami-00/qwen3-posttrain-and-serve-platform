terraform {
  required_version = "= 1.16.6"
  required_providers {
    aws = {
      source = "hashicorp/aws", version = "6.69.1"
    }
  }
}
variable "account_id" {
  type = string
}
variable "region" {
  type = string
}
variable "prefix" {
  type = string
}
variable "buckets" {
  type = map(string)
}
variable "secrets" {
  type = map(string)
}
variable "provision_role_arn" {
  type = string
}
variable "source_key" {
  type = string
}
variable "source_sha" {
  type = string
}
provider "aws" {
  region              = var.region
  allowed_account_ids = [var.account_id]
  default_tags {
    tags = {
      Project = "qps", Environment = "test", ManagedBy = "terraform"
    }
  }
}
module "network" {
  source = "../../../core/modules/aws/network"
  prefix = var.prefix
  region = var.region
}
resource "aws_security_group" "operations" {
  name        = "${var.prefix}-operations"
  description = "Private cluster and database operations"
  vpc_id      = module.network.vpc_id
}
resource "aws_vpc_security_group_egress_rule" "operations" {
  security_group_id = aws_security_group.operations.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}
module "eks" {
  source              = "../../../core/modules/aws/eks"
  prefix              = var.prefix
  region              = var.region
  account_id          = var.account_id
  subnet_ids          = module.network.private_subnet_ids
  provision_role_arn  = var.provision_role_arn
  artifact_bucket_arn = "arn:aws:s3:::${var.buckets["artifacts"]}"
  logs_bucket_arn     = "arn:aws:s3:::${var.buckets["logs"]}"
  api_secret_arn      = var.secrets["api"]
  operator_secret_arn = var.secrets["operator"]
}
resource "aws_vpc_security_group_ingress_rule" "private_api" {
  security_group_id            = module.eks.security_group_id
  referenced_security_group_id = aws_security_group.operations.id
  ip_protocol                  = "tcp"
  from_port                    = 443
  to_port                      = 443
}
module "postgres" {
  source     = "../../../core/modules/aws/postgres"
  prefix     = var.prefix
  vpc_id     = module.network.vpc_id
  subnet_ids = module.network.private_subnet_ids
  client_security_group_ids = {
    nodes = module.eks.security_group_id, operations = aws_security_group.operations.id
  }
}
module "edge" {
  source                 = "../../../core/modules/aws/edge"
  prefix                 = var.prefix
  account_id             = var.account_id
  region                 = var.region
  vpc_id                 = module.network.vpc_id
  subnet_ids             = module.network.private_subnet_ids
  node_security_group_id = module.eks.security_group_id
}
resource "aws_iam_role_policy" "database_operations" {
  role = split("/", var.provision_role_arn)[1]
  policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = module.postgres.master_secret_arn
    }]
  })
}
resource "aws_codebuild_project" "cluster" {
  name                   = "${var.prefix}-cluster"
  service_role           = var.provision_role_arn
  build_timeout          = 40
  queued_timeout         = 5
  concurrent_build_limit = 1
  artifacts {
    type = "NO_ARTIFACTS"
  }
  source {
    type      = "S3"
    location  = "${var.buckets["sources"]}/${var.source_key}"
    buildspec = "containers/operations.yml"
  }
  environment {
    compute_type = "BUILD_GENERAL1_SMALL"
    image        = "aws/codebuild/standard:8.0"
    type         = "LINUX_CONTAINER"
    environment_variable {
      name  = "QPS_OPERATION"
      value = "cluster-apply"
    }
    environment_variable {
      name  = "QPS_ACCOUNT_ID"
      value = var.account_id
    }
    environment_variable {
      name  = "QPS_REGION"
      value = var.region
    }
    environment_variable {
      name  = "QPS_PREFIX"
      value = var.prefix
    }
    environment_variable {
      name  = "QPS_SOURCE_SHA"
      value = var.source_sha
    }
    environment_variable {
      name  = "QPS_STATE_BUCKET"
      value = var.buckets["state"]
    }
    environment_variable {
      name  = "QPS_REPORTS_BUCKET"
      value = var.buckets["reports"]
    }
  }
  vpc_config {
    vpc_id             = module.network.vpc_id
    subnets            = module.network.private_subnet_ids
    security_group_ids = [aws_security_group.operations.id]
  }
  logs_config {
    cloudwatch_logs {
      group_name  = "/qps/test/build"
      stream_name = "cluster"
    }
  }
}
output "stack" {
  value = {
    url                    = module.edge.url
    cluster_name           = module.eks.name
    cluster_endpoint       = module.eks.endpoint
    cluster_ca             = module.eks.ca
    vpc_id                 = module.network.vpc_id
    target_groups          = module.edge.target_groups
    database_host          = module.postgres.host
    database_master_secret = module.postgres.master_secret_arn
    cognito_pool_id        = module.edge.cognito_pool_id
    cognito_client_id      = module.edge.cognito_client_id
    cognito_issuer         = module.edge.cognito_issuer
    cognito_domain         = module.edge.cognito_domain
    cluster_project        = aws_codebuild_project.cluster.name
  }
}
