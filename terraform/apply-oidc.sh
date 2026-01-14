#!/bin/bash
set -e

echo "--- Iniciando unificación de infraestructura con Terraform ---"

cd terraform

# Inicializar terraform
terraform init

# Ejecutar el plan
echo "--- Generando plan de ejecución ---"
terraform plan -var "account_id=${TF_VAR_account_id:-510701314494}" -var "aws_region=${TF_VAR_aws_region:-us-east-1}"

# Aplicar (usando -auto-approve para automatización en este entorno)
echo "--- Aplicando cambios ---"
terraform apply -auto-approve -var "account_id=${TF_VAR_account_id:-510701314494}" -var "aws_region=${TF_VAR_aws_region:-us-east-1}"
