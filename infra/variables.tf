variable "aws_region" {
  description = "AWS region for all resources. Interview account is provisioned in us-east-1."
  type        = string
  default     = "us-east-1"
}

variable "candidate_suffix" {
  description = "Unique per-candidate prefix applied to every resource name so parallel candidates in the shared account do not collide."
  type        = string
  default     = "yen-churn"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,30}[a-z0-9]$", var.candidate_suffix))
    error_message = "candidate_suffix must be lowercase alphanumeric with hyphens (S3 bucket naming rules)."
  }
}

variable "environment" {
  description = "Deployment environment tag."
  type        = string
  default     = "interview"
}
