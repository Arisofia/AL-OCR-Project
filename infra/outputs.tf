output "s3_bucket" {
  value = aws_s3_bucket.documents.id
}

output "ecr_repo" {
  value = aws_ecr_repository.ocr_service.repository_url
}