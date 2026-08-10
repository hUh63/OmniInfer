#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/../vla.cpp-linux/build.sh"
export OMNIINFER_VLA_CPP_BACKEND_ID="vla.cpp-linux-cuda"
export OMNIINFER_VLA_CPP_BACKEND_LABEL="vla.cpp Linux CUDA"
export OMNIINFER_VLA_CPP_ENABLE_CUDA=1
exec bash "${INNER_SCRIPT}" "$@"
