terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state only. A remote S3 backend would need a pre-existing bucket +
  # lock table, which is not worth the spend on a 4-day shared account.
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "churn-prediction"
      Candidate   = var.candidate_suffix
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  name        = var.candidate_suffix
  bucket_name = "${var.candidate_suffix}-artifacts"
  models_dir  = "${path.module}/../models"

  # Local file -> S3 key. Only the artifacts the scoring service loads at
  # startup; SHAP plots and the joblib baseline stay out of the runtime path.
  model_artifacts = {
    "churn_model.json"  = "artifacts/churn_model.json"
    "feature_spec.json" = "artifacts/feature_spec.json"
    "threshold.json"    = "artifacts/threshold.json"
    "model_card.json"   = "artifacts/model_card.json"
    "metrics.json"      = "artifacts/metrics.json"
  }
}

# ---------------------------------------------------------------------------
# S3: model artifact store
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "artifacts" {
  bucket = local.bucket_name

  # Interview account teardown: `terraform destroy` must not strand versioned
  # objects.
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "model_artifact" {
  for_each = local.model_artifacts

  bucket       = aws_s3_bucket.artifacts.id
  key          = each.value
  source       = "${local.models_dir}/${each.key}"
  content_type = "application/json"

  # Re-uploads when the local artifact changes; without this Terraform only
  # tracks the key, not the bytes.
  etag = filemd5("${local.models_dir}/${each.key}")

  depends_on = [aws_s3_bucket_versioning.artifacts]
}

# ---------------------------------------------------------------------------
# ECR: prediction service image
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "service" {
  name                 = "${local.name}-service"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  # Allows destroy while tagged images are still present.
  force_delete = true
}

# Keep only the last 5 images so the shared account does not accumulate
# storage charges across candidates.
resource "aws_ecr_lifecycle_policy" "service" {
  repository = aws_ecr_repository.service.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire all but the 5 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# ---------------------------------------------------------------------------
# CloudWatch: logs, error metric, alarm
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "service" {
  name              = "/${local.name}/prediction-service"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_metric_filter" "errors" {
  name           = "${local.name}-errors"
  log_group_name = aws_cloudwatch_log_group.service.name
  pattern        = "ERROR"

  metric_transformation {
    name      = "ChurnServiceErrors"
    namespace = "YenChurn"
    value     = "1"

    # Emits 0 on non-matching log events so the alarm has data to evaluate
    # while the service is healthy.
    default_value = "0"
  }
}

resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "${local.name}-service-errors"
  alarm_description   = "5 or more ERROR log lines from the churn prediction service within 5 minutes."
  namespace           = "YenChurn"
  metric_name         = "ChurnServiceErrors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanOrEqualToThreshold"

  # No metric data exists until the service logs its first line; absence of
  # errors must not read as an alarm.
  treat_missing_data = "notBreaching"

  # No SNS action wired up: notification targets are out of scope and an
  # unconfirmed subscription would just be dead config.
  alarm_actions = []

  depends_on = [aws_cloudwatch_log_metric_filter.errors]
}

# ---------------------------------------------------------------------------
# IAM: least-privilege artifact read role for the service
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "pod_trust" {
  statement {
    sid     = "EC2AssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "pod" {
  name               = "${local.name}-pod-role"
  description        = "Read-only access to churn model artifacts for the prediction service."
  assume_role_policy = data.aws_iam_policy_document.pod_trust.json
}

data "aws_iam_policy_document" "artifact_read" {
  statement {
    sid       = "ReadModelArtifacts"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/artifacts/*"]
  }

  statement {
    sid       = "ListArtifactPrefix"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["artifacts/*"]
    }
  }
}

resource "aws_iam_role_policy" "artifact_read" {
  name   = "${local.name}-artifact-read"
  role   = aws_iam_role.pod.id
  policy = data.aws_iam_policy_document.artifact_read.json
}
