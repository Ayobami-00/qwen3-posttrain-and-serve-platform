terraform {
  required_version = "= 1.16.6"
  required_providers {
    aws = {
      source = "hashicorp/aws", version = "6.69.1"
    }
    archive = {
      source = "hashicorp/archive", version = "2.8.0"
    }
  }
}
provider "aws" {
  region              = var.region
  allowed_account_ids = [var.account_id]
  default_tags {
    tags = local.tags
  }
}
locals {
  prefix = "${var.project}-${var.environment}"
  tags = {
    Project = var.project, Environment = var.environment, ManagedBy = "terraform"
  }
  buckets = toset(["state", "artifacts", "reports", "sources", "logs"])
}
resource "aws_s3_bucket" "project" {
  for_each      = local.buckets
  bucket        = "${local.prefix}-${var.account_id}-${var.region}-${each.key}"
  force_destroy = false
}
resource "aws_s3_bucket_public_access_block" "project" {
  for_each                = aws_s3_bucket.project
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_ownership_controls" "project" {
  for_each = aws_s3_bucket.project
  bucket   = each.value.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}
resource "aws_s3_bucket_versioning" "project" {
  for_each = aws_s3_bucket.project
  bucket   = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "project" {
  for_each = aws_s3_bucket.project
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
resource "aws_s3_bucket_policy" "tls" {
  for_each = aws_s3_bucket.project
  bucket   = each.value.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Deny", Principal = "*", Action = "s3:*", Resource = [each.value.arn, "${each.value.arn}/*"], Condition = {
        Bool = {
          "aws:SecureTransport" = "false"
        }
      }
    }]
  })
}
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.project["artifacts"].id
  rule {
    id     = "weights-seven-days"
    status = "Enabled"
    filter {
      tag {
        key   = "qps-retention"
        value = "weights"
      }
    }
    expiration {
      days = 7
    }
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
  rule {
    id     = "reports-thirty-days"
    status = "Enabled"
    filter {
      tag {
        key   = "qps-retention"
        value = "reports"
      }
    }
    expiration {
      days = 30
    }
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
  rule {
    id     = "staging-one-day"
    status = "Enabled"
    filter {
      prefix = "${var.environment}/staging/"
    }
    expiration {
      days = 1
    }
    noncurrent_version_expiration {
      noncurrent_days = 1
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
  rule {
    id     = "allocation-ledger"
    status = "Enabled"
    filter {
      prefix = "${var.environment}/allocations/"
    }
    expiration {
      days = 30
    }
    noncurrent_version_expiration {
      noncurrent_days = 1
    }
  }
}
resource "aws_s3_bucket_lifecycle_configuration" "ephemeral" {
  for_each = toset(["reports", "sources", "logs"])
  bucket   = aws_s3_bucket.project[each.key].id
  rule {
    id     = "bounded-retention"
    status = "Enabled"
    filter {
    }
    expiration {
      days = each.key == "reports" ? 30 : 7
    }
    noncurrent_version_expiration {
      noncurrent_days = each.key == "reports" ? 30 : 7
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}
resource "aws_ecr_repository" "images" {
  for_each             = toset(["api", "worker", "worker-cpu", "web", "operator"])
  name                 = "${local.prefix}/${each.key}"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = false
  }
}
resource "aws_ecr_lifecycle_policy" "images" {
  for_each   = aws_ecr_repository.images
  repository = each.value.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1, description = "Expire test build images after seven days", selection = {
        tagStatus = "any", countType = "sinceImagePushed", countUnit = "days", countNumber = 7
        }, action = {
        type = "expire"
      }
    }]
  })
}
resource "aws_secretsmanager_secret" "credentials" {
  for_each                = toset(["api", "operator", "janitor", "owner"])
  name                    = "${local.prefix}/${each.key}"
  recovery_window_in_days = 0
}
resource "aws_cloudwatch_log_group" "build" {
  name              = "/qps/${var.environment}/build"
  retention_in_days = 1
}
resource "aws_iam_role" "build" {
  name = "${local.prefix}-build"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = {
        Service = "codebuild.amazonaws.com"
        }, Action = "sts:AssumeRole", Condition = {
        StringEquals = {
          "aws:SourceAccount" = var.account_id
          }, ArnLike = {
          "aws:SourceArn" = "arn:aws:codebuild:${var.region}:${var.account_id}:project/${local.prefix}-*"
        }
      }
    }]
  })
}
resource "aws_iam_role_policy" "build" {
  role = aws_iam_role.build.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:GetBucketLocation", "s3:GetBucketAcl"], Resource = flatten([for key in ["sources", "reports"] : [aws_s3_bucket.project[key].arn, "${aws_s3_bucket.project[key].arn}/*"]])
      },
      {
        Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*"
      },
      {
        Effect = "Allow", Action = ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:DescribeImages"], Resource = [for image in aws_ecr_repository.images : image.arn]
      },
      {
        Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.build.arn}:*"
      }
    ]
  })
}
resource "aws_iam_role" "provision" {
  name               = "${local.prefix}-provision"
  assume_role_policy = aws_iam_role.build.assume_role_policy
}
resource "aws_iam_role_policy" "provision" {
  role = aws_iam_role.provision.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect = "Allow", Action = ["ec2:Describe*", "eks:Describe*", "eks:List*", "rds:Describe*", "iam:Get*", "iam:List*", "elasticloadbalancing:Describe*", "cloudfront:Get*", "cloudfront:List*", "kms:DescribeKey", "pricing:GetProducts", "servicequotas:Get*"], Resource = "*"
      },
      {
        Effect = "Allow", Action = ["ec2:CreateVpc", "ec2:DeleteVpc", "ec2:ModifyVpcAttribute", "ec2:CreateSubnet", "ec2:DeleteSubnet", "ec2:ModifySubnetAttribute", "ec2:CreateRouteTable", "ec2:DeleteRouteTable", "ec2:CreateRoute", "ec2:DeleteRoute", "ec2:ReplaceRoute", "ec2:AssociateRouteTable", "ec2:DisassociateRouteTable", "ec2:CreateInternetGateway", "ec2:DeleteInternetGateway", "ec2:AttachInternetGateway", "ec2:DetachInternetGateway", "ec2:AllocateAddress", "ec2:ReleaseAddress", "ec2:CreateNatGateway", "ec2:DeleteNatGateway", "ec2:CreateSecurityGroup", "ec2:DeleteSecurityGroup", "ec2:AuthorizeSecurityGroupIngress", "ec2:AuthorizeSecurityGroupEgress", "ec2:RevokeSecurityGroupIngress", "ec2:RevokeSecurityGroupEgress", "ec2:CreateTags", "ec2:DeleteTags", "ec2:CreateVpcEndpoint", "ec2:DeleteVpcEndpoints", "ec2:ModifyVpcEndpoint", "ec2:CreateNetworkInterface", "ec2:DeleteNetworkInterface", "ec2:CreateNetworkInterfacePermission", "ec2:CreateLaunchTemplate", "ec2:DeleteLaunchTemplate", "ec2:CreateLaunchTemplateVersion", "ec2:DeleteLaunchTemplateVersions", "ec2:ModifyLaunchTemplate"], Resource = "*"
      },
      {
        Effect = "Allow", Action = ["eks:CreateCluster", "eks:DeleteCluster", "eks:UpdateClusterConfig", "eks:CreateNodegroup", "eks:DeleteNodegroup", "eks:UpdateNodegroupConfig", "eks:CreateAddon", "eks:DeleteAddon", "eks:UpdateAddon", "eks:CreateAccessEntry", "eks:DeleteAccessEntry", "eks:UpdateAccessEntry", "eks:AssociateAccessPolicy", "eks:DisassociateAccessPolicy", "eks:CreatePodIdentityAssociation", "eks:DeletePodIdentityAssociation", "eks:UpdatePodIdentityAssociation", "eks:TagResource", "eks:UntagResource"], Resource = "arn:aws:eks:${var.region}:${var.account_id}:*"
      },
      {
        Effect = "Allow", Action = ["rds:CreateDBInstance", "rds:DeleteDBInstance", "rds:ModifyDBInstance", "rds:CreateDBSubnetGroup", "rds:DeleteDBSubnetGroup", "rds:ModifyDBSubnetGroup", "rds:AddTagsToResource", "rds:RemoveTagsFromResource", "rds:ListTagsForResource"], Resource = "arn:aws:rds:${var.region}:${var.account_id}:*:${local.prefix}*"
      },
      {
        Effect = "Allow", Action = ["elasticloadbalancing:Create*", "elasticloadbalancing:Delete*", "elasticloadbalancing:Modify*", "elasticloadbalancing:AddTags", "elasticloadbalancing:RemoveTags", "elasticloadbalancing:Set*"], Resource = "*"
      },
      {
        Effect = "Allow", Action = ["cloudfront:CreateDistribution", "cloudfront:UpdateDistribution", "cloudfront:DeleteDistribution", "cloudfront:CreateVpcOrigin", "cloudfront:UpdateVpcOrigin", "cloudfront:DeleteVpcOrigin", "cloudfront:TagResource", "cloudfront:UntagResource"], Resource = "*"
      },
      {
        Effect = "Allow", Action = ["iam:CreateRole", "iam:DeleteRole", "iam:UpdateRole", "iam:UpdateAssumeRolePolicy", "iam:PutRolePolicy", "iam:DeleteRolePolicy", "iam:AttachRolePolicy", "iam:DetachRolePolicy", "iam:TagRole", "iam:UntagRole", "iam:PassRole"], Resource = "arn:aws:iam::${var.account_id}:role/${local.prefix}-*"
      },
      {
        Effect = "Allow", Action = ["iam:CreateServiceLinkedRole"], Resource = "arn:aws:iam::${var.account_id}:role/aws-service-role/*", Condition = {
          StringEquals = {
            "iam:AWSServiceName" = ["eks.amazonaws.com", "eks-nodegroup.amazonaws.com", "elasticloadbalancing.amazonaws.com", "cloudfront.amazonaws.com", "vpcorigin.cloudfront.amazonaws.com", "rds.amazonaws.com"]
          }
        }
      },
      {
        Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:GetObjectVersion", "s3:ListBucket", "s3:GetBucketLocation"], Resource = flatten([for bucket in aws_s3_bucket.project : [bucket.arn, "${bucket.arn}/*"]])
      },
      {
        Effect = "Allow", Action = ["s3:CreateBucket", "s3:DeleteBucket", "s3:Get*", "s3:List*", "s3:PutBucket*", "s3:DeleteBucket*"], Resource = "arn:aws:s3:::${local.prefix}-*"
      },
      {
        Effect = "Allow", Action = ["secretsmanager:GetSecretValue", "secretsmanager:PutSecretValue", "secretsmanager:DescribeSecret", "secretsmanager:CreateSecret", "secretsmanager:TagResource"], Resource = ["arn:aws:secretsmanager:${var.region}:${var.account_id}:secret:${local.prefix}/*", "arn:aws:secretsmanager:${var.region}:${var.account_id}:secret:rds!*"]
      },
      {
        Effect = "Allow", Action = ["codebuild:CreateProject", "codebuild:UpdateProject", "codebuild:DeleteProject", "codebuild:BatchGetProjects", "codebuild:BatchGetBuilds", "codebuild:StartBuild"], Resource = ["arn:aws:codebuild:${var.region}:${var.account_id}:project/${local.prefix}-*", "arn:aws:codebuild:${var.region}:${var.account_id}:build/${local.prefix}-*"]
      },
      {
        Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:DeleteLogGroup", "logs:PutRetentionPolicy", "logs:DeleteRetentionPolicy", "logs:TagResource", "logs:UntagResource", "logs:DescribeLogGroups", "logs:ListTagsForResource", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "*"
      },
      {
        Effect = "Allow", Action = ["cognito-idp:GetUserPoolMfaConfig", "cognito-idp:SetUserPoolMfaConfig", "cognito-idp:CreateUserPool", "cognito-idp:DeleteUserPool", "cognito-idp:UpdateUserPool", "cognito-idp:DescribeUserPool", "cognito-idp:CreateUserPoolClient", "cognito-idp:UpdateUserPoolClient", "cognito-idp:DeleteUserPoolClient", "cognito-idp:DescribeUserPoolClient", "cognito-idp:CreateUserPoolDomain", "cognito-idp:DeleteUserPoolDomain", "cognito-idp:DescribeUserPoolDomain", "cognito-idp:CreateGroup", "cognito-idp:DeleteGroup", "cognito-idp:GetGroup", "cognito-idp:AdminCreateUser", "cognito-idp:AdminDeleteUser", "cognito-idp:AdminGetUser", "cognito-idp:AdminSetUserPassword", "cognito-idp:AdminAddUserToGroup", "cognito-idp:AdminRemoveUserFromGroup", "cognito-idp:AdminListGroupsForUser", "cognito-idp:ListTagsForResource", "cognito-idp:TagResource", "cognito-idp:UntagResource"], Resource = "*"
      }
    ]
  })
}
resource "aws_iam_role_policy" "provision_build" {
  role   = aws_iam_role.provision.id
  policy = aws_iam_role_policy.build.policy
}
resource "aws_codebuild_project" "build" {
  name                   = "${local.prefix}-images"
  service_role           = aws_iam_role.build.arn
  build_timeout          = 60
  queued_timeout         = 5
  concurrent_build_limit = 1
  artifacts {
    type = "NO_ARTIFACTS"
  }
  source {
    type      = "S3"
    location  = "${aws_s3_bucket.project["sources"].id}/${var.source_key}"
    buildspec = "containers/buildspec.yml"
  }
  environment {
    compute_type    = "BUILD_GENERAL1_MEDIUM"
    image           = "aws/codebuild/standard:8.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true
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
      value = local.prefix
    }
    environment_variable {
      name  = "QPS_SOURCE_SHA"
      value = var.source_sha
    }
    environment_variable {
      name  = "QPS_REPORTS_BUCKET"
      value = aws_s3_bucket.project["reports"].id
    }
  }
  logs_config {
    cloudwatch_logs {
      group_name  = aws_cloudwatch_log_group.build.name
      stream_name = "images"
    }
  }
}
resource "aws_codebuild_project" "operations" {
  name                   = "${local.prefix}-operations"
  service_role           = aws_iam_role.provision.arn
  build_timeout          = 45
  queued_timeout         = 5
  concurrent_build_limit = 1
  artifacts {
    type = "NO_ARTIFACTS"
  }
  source {
    type      = "S3"
    location  = "${aws_s3_bucket.project["sources"].id}/${var.source_key}"
    buildspec = "containers/operations.yml"
  }
  environment {
    compute_type = "BUILD_GENERAL1_SMALL"
    image        = "aws/codebuild/standard:8.0"
    type         = "LINUX_CONTAINER"
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
      value = local.prefix
    }
    environment_variable {
      name  = "QPS_SOURCE_SHA"
      value = var.source_sha
    }
    environment_variable {
      name  = "QPS_REPORTS_BUCKET"
      value = aws_s3_bucket.project["reports"].id
    }
    environment_variable {
      name  = "QPS_STATE_BUCKET"
      value = aws_s3_bucket.project["state"].id
    }
    environment_variable {
      name  = "QPS_OPERATION"
      value = "status"
    }
  }
  logs_config {
    cloudwatch_logs {
      group_name  = aws_cloudwatch_log_group.build.name
      stream_name = "operations"
    }
  }
}
resource "aws_iam_role" "janitor" {
  name = "${local.prefix}-janitor"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = {
        Service = "lambda.amazonaws.com"
      }, Action = "sts:AssumeRole"
    }]
  })
}
resource "aws_cloudwatch_log_group" "janitor" {
  name              = "/aws/lambda/${local.prefix}-janitor"
  retention_in_days = 1
}
resource "aws_iam_role_policy" "janitor" {
  role = aws_iam_role.janitor.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect = "Allow", Action = ["s3:ListBucket"], Resource = [aws_s3_bucket.project["artifacts"].arn, aws_s3_bucket.project["reports"].arn]
      },
      {
        Effect = "Allow", Action = ["s3:GetObject"], Resource = ["${aws_s3_bucket.project["artifacts"].arn}/${var.environment}/allocations/*", "${aws_s3_bucket.project["reports"].arn}/installation/*"]
      },
      {
        Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${aws_s3_bucket.project["reports"].arn}/cleanup/*"
      },
      {
        Effect = "Allow", Action = "s3:PutObject", Resource = "${aws_s3_bucket.project["reports"].arn}/installation/current.json"
      },
      {
        Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = aws_secretsmanager_secret.credentials["janitor"].arn
      },
      {
        Effect = "Allow", Action = ["codebuild:StartBuild", "codebuild:BatchGetBuilds"], Resource = [aws_codebuild_project.operations.arn, "arn:aws:codebuild:${var.region}:${var.account_id}:build/${local.prefix}-operations:*"]
      },
      {
        Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.janitor.arn}:*"
      }
    ]
  })
}
data "archive_file" "janitor" {
  type        = "zip"
  source_file = "${path.module}/../../lambda/janitor.py"
  output_path = "${path.module}/janitor.zip"
}
resource "aws_lambda_function" "janitor" {
  function_name                  = "${local.prefix}-janitor"
  role                           = aws_iam_role.janitor.arn
  runtime                        = "python3.14"
  handler                        = "janitor.handler"
  filename                       = data.archive_file.janitor.output_path
  source_code_hash               = data.archive_file.janitor.output_base64sha256
  timeout                        = 120
  memory_size                    = 128
  reserved_concurrent_executions = 1
  environment {
    variables = {
      ARTIFACT_BUCKET = aws_s3_bucket.project["artifacts"].id, REPORT_BUCKET = aws_s3_bucket.project["reports"].id, SECRET_ARN = aws_secretsmanager_secret.credentials["janitor"].arn, PREFIX = local.prefix, ENVIRONMENT = var.environment, EXPIRY = var.expires_at, OPERATIONS_PROJECT = aws_codebuild_project.operations.name
    }
  }
  depends_on = [aws_cloudwatch_log_group.janitor]
}
resource "aws_cloudwatch_event_rule" "janitor" {
  name                = "${local.prefix}-janitor"
  schedule_expression = "rate(1 minute)"
}
resource "aws_cloudwatch_event_target" "janitor" {
  rule = aws_cloudwatch_event_rule.janitor.name
  arn  = aws_lambda_function.janitor.arn
}
resource "aws_lambda_permission" "janitor" {
  statement_id  = "AllowOwnedSchedule"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.janitor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.janitor.arn
}

resource "aws_iam_role_policy" "invoke_cleanup" {
  role   = aws_iam_role.provision.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "lambda:InvokeFunction", Resource = aws_lambda_function.janitor.arn }] })
}

resource "aws_iam_role_policy" "close_credentials" {
  role = aws_iam_role.provision.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = "secretsmanager:DeleteSecret", Resource = [for secret in aws_secretsmanager_secret.credentials : secret.arn] },
    { Effect = "Allow", Action = "events:DisableRule", Resource = aws_cloudwatch_event_rule.janitor.arn }
  ] })
}

resource "aws_iam_role" "expiry" {
  name               = "${local.prefix}-expiry"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "scheduler.amazonaws.com" }, Action = "sts:AssumeRole", Condition = { StringEquals = { "aws:SourceAccount" = var.account_id } } }] })
}

resource "aws_iam_role_policy" "expiry" {
  role   = aws_iam_role.expiry.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "codebuild:StartBuild", Resource = aws_codebuild_project.operations.arn }] })
}

resource "aws_scheduler_schedule" "expiry" {
  name                         = "${local.prefix}-expiry"
  schedule_expression          = "at(${formatdate("YYYY-MM-DD'T'hh:mm:ss", var.expires_at)})"
  schedule_expression_timezone = "UTC"
  flexible_time_window { mode = "OFF" }
  target {
    arn      = "arn:aws:scheduler:::aws-sdk:codebuild:startBuild"
    role_arn = aws_iam_role.expiry.arn
    input    = jsonencode({ ProjectName = aws_codebuild_project.operations.name, EnvironmentVariablesOverride = [{ Name = "QPS_OPERATION", Value = "destroy", Type = "PLAINTEXT" }] })
    retry_policy {
      maximum_event_age_in_seconds = 3600
      maximum_retry_attempts       = 3
    }
  }
}
