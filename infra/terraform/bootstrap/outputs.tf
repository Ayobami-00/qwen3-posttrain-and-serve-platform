output "buckets" {
  value = {
    for key, bucket in aws_s3_bucket.project : key => bucket.id
  }
}
output "repositories" {
  value = {
    for key, repo in aws_ecr_repository.images : key => repo.repository_url
  }
}
output "secrets" {
  value = {
    for key, secret in aws_secretsmanager_secret.credentials : key => secret.arn
  }
}
output "build_project" {
  value = aws_codebuild_project.build.name
}
output "operations_project" {
  value = aws_codebuild_project.operations.name
}
output "provision_role_arn" {
  value = aws_iam_role.provision.arn
}
output "janitor_arn" {
  value = aws_lambda_function.janitor.arn
}
