#!/usr/bin/env bash
# Deploy SkillGPT to Azure Container Apps.
#
#   ./deploy/azure/deploy.sh <resource-group> <region> <registry>.azurecr.io/<image>:<tag>
#
# Expects to be logged in (`az login`) and the image already pushed. Keys are read from
# the environment so none of them ever appears in a shell history or a command line:
#   OPEN_API_KEY  GEMINI_API_KEY  ZILLIZ_URI  ZILLIZ_TOKEN
#   REGISTRY_USER  REGISTRY_PASSWORD

set -euo pipefail

RG="${1:?resource group required}"
LOCATION="${2:?location required}"
IMAGE="${3:?container image required}"
REGISTRY="${IMAGE%%/*}"

az group create --name "$RG" --location "$LOCATION" --output none

az deployment group create \
  --resource-group "$RG" \
  --template-file "$(dirname "$0")/main.bicep" \
  --parameters \
      containerImage="$IMAGE" \
      registryServer="$REGISTRY" \
      registryUsername="${REGISTRY_USER:?}" \
      registryPassword="${REGISTRY_PASSWORD:?}" \
      openAiKey="${OPEN_API_KEY:-}" \
      geminiKey="${GEMINI_API_KEY:-}" \
      zillizUri="${ZILLIZ_URI:-}" \
      zillizToken="${ZILLIZ_TOKEN:-}" \
  --output table

URL=$(az deployment group show -g "$RG" -n main --query properties.outputs.apiUrl.value -o tsv)
echo "deployed: $URL"

# Verify rather than assume: a template that applied cleanly can still produce a
# container that does not serve.
for _ in $(seq 1 30); do
  if curl -fs "$URL/api/health"; then echo; echo "  healthy"; exit 0; fi
  sleep 5
done
echo "  the service did not become healthy" >&2
exit 1
