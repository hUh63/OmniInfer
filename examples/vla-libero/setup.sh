#!/usr/bin/env bash
# Create the small runtime needed by the interactive dashboard, without
# replacing vla.cpp's complete (and much larger) LIBERO evaluation environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VLA_CPP_ROOT="$REPOSITORY_ROOT/framework/vla.cpp"
DEMO_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/omniinfer/vla-libero-demo"
VENV_DIR="$DEMO_CACHE_DIR/venv"
LIBERO_DIR="$DEMO_CACHE_DIR/LIBERO"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
TORCH_BACKEND="cpu"
UV_BIN="${UV_BIN:-}"
RUN_SMOKE_TEST=0

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --torch-backend <backend>  uv PyTorch backend (default: cpu; e.g. cu124)
  --venv <path>              virtual environment directory
  --libero-dir <path>        LIBERO source checkout directory
  --uv <path>                uv executable (default: uv from PATH)
  --smoke-test               Reset one LIBERO task after setup (default: off)
  -h, --help                 show this help
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
        --torch-backend) require_value "$1" "${2-}"; TORCH_BACKEND="$2"; shift 2 ;;
        --venv) require_value "$1" "${2-}"; VENV_DIR="$2"; shift 2 ;;
        --libero-dir) require_value "$1" "${2-}"; LIBERO_DIR="$2"; shift 2 ;;
        --uv) require_value "$1" "${2-}"; UV_BIN="$2"; shift 2 ;;
        --smoke-test) RUN_SMOKE_TEST=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$(uname -s)" == "Linux" ]] || {
    echo "this example currently supports Linux only" >&2
    exit 1
}

if [[ -z "$UV_BIN" ]]; then
    UV_BIN="$(command -v uv 2>/dev/null || true)"
    if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
        UV_BIN="$HOME/.local/bin/uv"
    fi
fi
[[ -n "$UV_BIN" ]] && command -v "$UV_BIN" >/dev/null 2>&1 || {
    echo "uv is required; install uv or pass --uv <path>." >&2
    exit 1
}
[[ -f "$SCRIPT_DIR/requirements.txt" ]] || {
    echo "missing $SCRIPT_DIR/requirements.txt" >&2
    exit 1
}
[[ -d "$VLA_CPP_ROOT" ]] || {
    echo "framework/vla.cpp is not initialized; run git submodule update --init framework/vla.cpp" >&2
    exit 1
}

if [[ ! -d "$LIBERO_DIR/.git" ]]; then
    if [[ -e "$LIBERO_DIR" && -n "$(find "$LIBERO_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "LIBERO directory exists but is not a Git checkout: $LIBERO_DIR" >&2
        exit 1
    fi
    LIBERO_URL="$(git -C "$VLA_CPP_ROOT" config -f .gitmodules --get submodule.external_dependencies/LIBERO.url 2>/dev/null || true)"
    LIBERO_URL="${LIBERO_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
    mkdir -p "$(dirname "$LIBERO_DIR")"
    git init "$LIBERO_DIR"
    git -C "$LIBERO_DIR" remote add origin "$LIBERO_URL"
fi

if [[ -n "$(git -C "$LIBERO_DIR" status --porcelain)" ]]; then
    echo "LIBERO checkout has local changes; refusing to switch revisions: $LIBERO_DIR" >&2
    exit 1
fi
if ! git -C "$LIBERO_DIR" cat-file -e "$LIBERO_COMMIT^{commit}" 2>/dev/null; then
    git -C "$LIBERO_DIR" fetch --depth 1 origin "$LIBERO_COMMIT"
fi
git -C "$LIBERO_DIR" checkout --detach "$LIBERO_COMMIT"
[[ "$(git -C "$LIBERO_DIR" rev-parse HEAD)" == "$LIBERO_COMMIT" ]] || {
    echo "failed to pin LIBERO to $LIBERO_COMMIT" >&2
    exit 1
}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$UV_BIN" venv "$VENV_DIR" --python 3.10
fi
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
    --torch-backend "$TORCH_BACKEND" \
    --requirements "$SCRIPT_DIR/requirements.txt"

# LIBERO's current source layout is namespace-like, so its editable package
# metadata does not expose the top-level module reliably. A .pth file is
# deterministic and keeps the downloaded source outside version control.
SITE_PACKAGES="$($VENV_DIR/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$LIBERO_DIR" > "$SITE_PACKAGES/omniinfer_libero_source.pth"

# robosuite 1.4.0 enables a process-wide /tmp/robosuite.log by default. That
# path is unsafe on multi-user hosts: another user's file can make every
# subsequent import fail with PermissionError. Keep the upstream defaults in a
# venv-local private macro file and disable only file logging.
ROBOSUITE_DIR="$SITE_PACKAGES/robosuite"
[[ -f "$ROBOSUITE_DIR/macros.py" ]] || {
    echo "robosuite macros.py was not installed in $ROBOSUITE_DIR" >&2
    exit 1
}
"$VENV_DIR/bin/python" - "$ROBOSUITE_DIR/macros.py" "$ROBOSUITE_DIR/macros_private.py" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
needle = 'FILE_LOGGING_LEVEL = "DEBUG"'
if source.count(needle) != 1:
    raise SystemExit("unexpected robosuite FILE_LOGGING_LEVEL definition")
target = Path(sys.argv[2])
target.write_text(source.replace(needle, "FILE_LOGGING_LEVEL = None"), encoding="utf-8")
PY

CONFIG_DIR="$(dirname "$VENV_DIR")/libero-config"
mkdir -p "$CONFIG_DIR"
if [[ ! -f "$CONFIG_DIR/config.yaml" ]]; then
    printf 'N\n' | LIBERO_CONFIG_PATH="$CONFIG_DIR" "$VENV_DIR/bin/python" -c 'import libero.libero'
fi

# Optionally exercise the same import and environment-reset path used by
# demo.py. Keep this opt-in because installation itself should not require an
# EGL-capable device; users can select another supported MuJoCo renderer by
# setting MUJOCO_GL before invoking setup.
if [[ $RUN_SMOKE_TEST -eq 1 ]]; then
    LIBERO_CONFIG_PATH="$CONFIG_DIR" PYTHONPATH="$VLA_CPP_ROOT/eval" \
        MUJOCO_GL="${MUJOCO_GL:-egl}" "$VENV_DIR/bin/python" - <<'PY'
import gymnasium as gym
import sim.libero  # noqa: F401 - registers the LIBERO environments

environment = gym.make("libero_object/task_0", seed=0, video_fps=30)
try:
    environment.reset(seed=0)
finally:
    environment.close()
PY
fi

echo "Demo environment ready: $VENV_DIR"
echo "LIBERO revision: $LIBERO_COMMIT"
echo "Torch backend: $TORCH_BACKEND"
echo "Run: $SCRIPT_DIR/run.sh --venv $VENV_DIR -- <demo options>"
