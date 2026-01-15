#!/bin/bash
set -e

echo "--- Iniciando unificación de infraestructura con Terraform ---"

cd terraform

# Inicializar terraform
terraform init

# Resolve Account ID
RESOLVED_ACCOUNT_ID=${TF_VAR_account_id:-$AWS_ACCOUNT_ID}
if [ -z "$RESOLVED_ACCOUNT_ID" ]; then
    RESOLVED_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text 2>/dev/null || echo "510701314494")
fi

# Ejecutar el plan
echo "--- Generando plan de ejecución para Cuenta: $RESOLVED_ACCOUNT_ID ---"
terraform plan -var "account_id=$RESOLVED_ACCOUNT_ID" -var "aws_region=${TF_VAR_aws_region:-us-east-1}"

# Aplicar (usando -auto-approve para automatización en este entorno)
echo "--- Aplicando cambios ---"
terraform apply -auto-approve -var "account_id=$RESOLVED_ACCOUNT_ID" -var "aws_region=${TF_VAR_aws_region:-us-east-1}"
