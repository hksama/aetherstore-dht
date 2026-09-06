#!/usr/bin/env bash
# Generate random binary test data for prototype transfers.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <size_in_mb>" >&2
  exit 1
fi

SIZE_MB="$1"

if ! [[ "${SIZE_MB}" =~ ^[0-9]+$ ]] || [[ "${SIZE_MB}" -lt 1 ]]; then
  echo "Size must be a positive integer (megabytes)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEND_DIR="${AETHER_SEND_DIR:-${SCRIPT_DIR}/../send}"
OUTPUT="${SEND_DIR}/${SIZE_MB}M.bin"

mkdir -p "${SEND_DIR}"

echo "Writing ${SIZE_MB} MiB of random data to ${OUTPUT}"
dd if=/dev/urandom of="${OUTPUT}" bs=1m count="${SIZE_MB}" status=progress 2>/dev/null \
  || dd if=/dev/urandom of="${OUTPUT}" bs=1048576 count="${SIZE_MB}"

echo "Wrote ${OUTPUT}"
