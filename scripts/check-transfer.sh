#!/usr/bin/env bash
# Bit-for-bit compare a sent file and a received file.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <send_file> <recv_file>" >&2
  echo "  Paths are relative to send/ and recv/ at the project root." >&2
  echo "  Example: $0 1M.bin received.bin" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEND_DIR="${AETHER_SEND_DIR:-${SCRIPT_DIR}/../send}"
RECV_DIR="${AETHER_RECV_DIR:-${SCRIPT_DIR}/../recv}"

SEND_FILE="${SEND_DIR}/$1"
RECV_FILE="${RECV_DIR}/$2"

if [[ ! -f "${SEND_FILE}" ]]; then
  echo "Send file not found: ${SEND_FILE}" >&2
  exit 1
fi

if [[ ! -f "${RECV_FILE}" ]]; then
  echo "Recv file not found: ${RECV_FILE}" >&2
  exit 1
fi

SEND_SIZE="$(wc -c < "${SEND_FILE}" | tr -d ' ')"
RECV_SIZE="$(wc -c < "${RECV_FILE}" | tr -d ' ')"

if [[ "${SEND_SIZE}" != "${RECV_SIZE}" ]]; then
  echo "Size mismatch: send=${SEND_SIZE} bytes, recv=${RECV_SIZE} bytes" >&2
  exit 1
fi

if cmp -s "${SEND_FILE}" "${RECV_FILE}"; then
  echo "OK: files match bit-for-bit (${SEND_SIZE} bytes)"
  echo "  send: ${SEND_FILE}"
  echo "  recv: ${RECV_FILE}"
  exit 0
fi

echo "Mismatch: files differ (both ${SEND_SIZE} bytes, content differs)" >&2
cmp -l "${SEND_FILE}" "${RECV_FILE}" | head -5 >&2 || true
exit 1
