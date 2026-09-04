#!/usr/bin/env bash
# Generate a shared self-signed TLS certificate for local QUIC prototypes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="${AETHER_CERT_DIR:-${SCRIPT_DIR}/../prototypes/certs}"
CERT_FILE="${CERT_DIR}/cert.pem"
KEY_FILE="${CERT_DIR}/key.pem"

mkdir -p "${CERT_DIR}"

openssl req -x509 \
  -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -days 3650 \
  -nodes \
  -keyout "${KEY_FILE}" \
  -out "${CERT_FILE}" \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost" \
  -addext "basicConstraints=CA:FALSE" \
  -addext "keyUsage=digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth,clientAuth"

chmod 600 "${KEY_FILE}"
chmod 644 "${CERT_FILE}"

echo "Wrote certificate: ${CERT_FILE}"
echo "Wrote private key:  ${KEY_FILE}"
