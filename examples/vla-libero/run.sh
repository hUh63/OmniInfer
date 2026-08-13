#!/usr/bin/env bash
# Run the dashboard with the isolated LIBERO configuration created by setup.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/omniinfer/vla-libero-demo/venv"

usage() {
    cat <<EOF
Usage: $0 [--venv <path>] [--libero-config <path>] -- [demo options]
EOF
}

require_value() {
    local option="$1"
    local value="${2-}"
    if [[ -z "$value" || "$value" == -* ]]; then
        echo "$option requires a value" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) require_value "$1" "${2-}"; VENV_DIR="$2"; shift 2 ;;
        --libero-config) require_value "$1" "${2-}"; LIBERO_CONFIG_DIR="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) break ;;
    esac
done

[[ "$(uname -s)" == "Linux" ]] || {
    echo "this example currently supports Linux only" >&2
    exit 1
}

[[ -x "$VENV_DIR/bin/python" ]] || {
    echo "demo environment not found: $VENV_DIR; run $SCRIPT_DIR/setup.sh first." >&2
    exit 1
}
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR:-$(dirname "$VENV_DIR")/libero-config}"
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/demo.py" "$@"
