output "s3_bucket_name" {
  description = "Name of the model artifact bucket."
  value       = aws_s3_bucket.artifacts.id
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker build/tag/push of the prediction service image."
  value       = aws_ecr_repository.service.repository_url
}

output "cloudwatch_log_group" {
  description = "CloudWatch Logs group the prediction service writes to."
  value       = aws_cloudwatch_log_group.service.name
}

output "iam_role_arn" {
  description = "ARN of the read-only artifact role assumed by the prediction service."
  value       = aws_iam_role.pod.arn
}
