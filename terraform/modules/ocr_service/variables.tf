variable "aws_region" { type = string }
variable "account_id" { type = string }
variable "s3_bucket_name" { type = string }
variable "ecr_repository_name" { type = string }

# If set to true, Terraform will delete images in ECR when destroying or replacing the
# repository. This is destructive: enabling it will permanently remove images.
# Default is false to avoid accidental data loss. To opt-in, set this to true and
# ensure your CI/CD and backup processes are prepared.
variable "ecr_force_delete" {
  type    = bool
  default = false
}

# Make ECR image tag mutability configurable. Default to IMMUTABLE for best practice.
variable "ecr_image_tag_mutability" {
  type    = string
  default = "IMMUTABLE"
}
