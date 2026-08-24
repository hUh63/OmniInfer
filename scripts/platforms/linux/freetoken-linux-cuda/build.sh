#!/usr/bin/env bash

set -euo pipefail

FREETOKEN_VERSION="0.1.2"
FLASHINFER_VERSION="0.6.17"
PYTHON_VERSION="${OMNIINFER_FREETOKEN_PYTHON:-3.12}"
PACKAGE_INDEX_URL="${OMNIINFER_FREETOKEN_INDEX_URL:-}"
UV_VERSION="0.12.5"
RUNTIME_WHEEL="freetoken-0.1.2-cp312-cp312-manylinux_2_27_x86_64.whl"
RUNTIME_WHEEL_SHA256="993afeb4ef1ee3a1c5302b3c46dea6545d86f4a6facc3ea386985e02f4466a2f"
KERNEL_WHEEL="freetoken_kernel_cache-0.1.2+cu130-py3-none-linux_x86_64.whl"
KERNEL_WHEEL_SHA256="a401e8d0fb80405e99e120f20c43b207662110b2ba7a3b93c84e04887c120a4f"
UV_ARCHIVE="uv-x86_64-unknown-linux-gnu.tar.gz"
UV_ARCHIVE_SHA256="68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2"

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
RUNTIME_ROOT="${REPO_ROOT}/.local/runtime/linux"
ACTIVE_ROOT="${RUNTIME_ROOT}/freetoken-linux-cuda"
VERSIONS_ROOT="${RUNTIME_ROOT}/.freetoken-linux-cuda-versions"
CACHE_ROOT="${REPO_ROOT}/.local/cache/freetoken"
TOOLS_ROOT="${REPO_ROOT}/.local/tools/uv/${UV_VERSION}"
LOG_ROOT="${REPO_ROOT}/tmp/test_results/install/freetoken-linux-cuda"
DRY_RUN=0
CHECK_DEPS=0
SMOKE_TEST=0

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Installs the pinned FreeToken Linux CUDA runtime into OmniInfer's local runtime tree.

Options:
  --python <version>  uv Python version, defaults to 3.12
  --smoke-test        run the CUDA import self-check after installation
  --from-source       accepted by the source installer; installs pinned release wheels
  --check-deps        report host compatibility without installing
  --dry-run           print the planned installation without changing files
  -h, --help          show this help message

Requirements:
  Linux x86_64, NVIDIA driver branch R580 or newer, curl, tar, and sha256sum.
  No system Python, CUDA toolkit, sudo, or global package installation is required.

Environment:
  OMNIINFER_FREETOKEN_INDEX_URL  optional trusted PyPI-compatible mirror
EOF
}

while (($# > 0)); do
  case "$1" in
    --python)
      PYTHON_VERSION="${2:?missing value for --python}"
      shift 2
      ;;
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --from-source)
      shift
      ;;
    --check-deps)
      CHECK_DEPS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
  echo "FreeToken v${FREETOKEN_VERSION} is pinned to its CPython 3.12 release wheel." >&2
  exit 1
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    return 1
  fi
}

nvidia_driver_branch() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null \
    | sed -n '1{s/[[:space:]]//g;s/\..*//;p;}'
}

check_host() {
  local rc=0 branch=""
  for command in curl tar sha256sum nvidia-smi; do
    if require_command "${command}"; then
      printf 'ok|%s\n' "${command}"
    else
      rc=1
    fi
  done
  if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "FreeToken v${FREETOKEN_VERSION} requires Linux x86_64." >&2
    rc=1
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    branch="$(nvidia_driver_branch)"
    if [[ ! "${branch}" =~ ^[0-9]+$ || "${branch}" -lt 580 ]]; then
      echo "FreeToken CUDA 13 requires NVIDIA driver branch R580 or newer; found '${branch:-unknown}'." >&2
      rc=1
    else
      printf 'ok|nvidia-driver|%s\n' "${branch}"
    fi
  fi
  return "${rc}"
}

if [[ ${CHECK_DEPS} -eq 1 ]]; then
  check_host
  exit $?
fi

runtime_url="https://github.com/FlashML-org/FreeToken/releases/download/v${FREETOKEN_VERSION}/${RUNTIME_WHEEL}"
kernel_url="https://github.com/FlashML-org/FreeToken/releases/download/v${FREETOKEN_VERSION}/freetoken_kernel_cache-0.1.2%2Bcu130-py3-none-linux_x86_64.whl"
uv_url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_ARCHIVE}"

if [[ ${DRY_RUN} -eq 1 ]]; then
  cat <<EOF
FreeToken Linux CUDA installation plan
  version: ${FREETOKEN_VERSION}
  flashinfer: ${FLASHINFER_VERSION}
  runtime: ${ACTIVE_ROOT}
  python: ${PYTHON_VERSION}
  runtime wheel: ${runtime_url}
  kernel cache: ${kernel_url}
  uv: ${uv_url}
  package index: ${PACKAGE_INDEX_URL:-PyPI}
EOF
  exit 0
fi

check_host >/dev/null

mkdir -p "${CACHE_ROOT}" "${TOOLS_ROOT}" "${VERSIONS_ROOT}" "${LOG_ROOT}"

download_verified() {
  local url="$1" destination="$2" expected="$3"
  if [[ -f "${destination}" ]] && printf '%s  %s\n' "${expected}" "${destination}" | sha256sum -c - >/dev/null 2>&1; then
    return
  fi
  rm -f "${destination}"
  curl -fL --retry 5 --retry-delay 3 --connect-timeout 20 -C - \
    -o "${destination}.part" "${url}"
  if ! printf '%s  %s\n' "${expected}" "${destination}.part" | sha256sum -c -; then
    rm -f "${destination}.part"
    echo "Checksum verification failed for ${url}." >&2
    return 1
  fi
  mv "${destination}.part" "${destination}"
}

uv_bin="${TOOLS_ROOT}/uv"
if command -v uv >/dev/null 2>&1 && [[ "$(uv --version 2>/dev/null)" == "uv ${UV_VERSION}" ]]; then
  uv_bin="$(command -v uv)"
fi
if [[ ! -x "${uv_bin}" ]]; then
  uv_archive_path="${CACHE_ROOT}/${UV_ARCHIVE}"
  download_verified "${uv_url}" "${uv_archive_path}" "${UV_ARCHIVE_SHA256}"
  uv_stage="${TOOLS_ROOT}.stage.$$"
  rm -rf "${uv_stage}"
  mkdir -p "${uv_stage}"
  trap 'rm -rf "${uv_stage:-}" "${candidate:-}"' EXIT
  tar -xzf "${uv_archive_path}" --strip-components=1 -C "${uv_stage}"
  test -x "${uv_stage}/uv"
  rm -rf "${TOOLS_ROOT}"
  mv "${uv_stage}" "${TOOLS_ROOT}"
fi

runtime_wheel_path="${CACHE_ROOT}/${RUNTIME_WHEEL}"
kernel_wheel_path="${CACHE_ROOT}/${KERNEL_WHEEL}"
download_verified "${runtime_url}" "${runtime_wheel_path}" "${RUNTIME_WHEEL_SHA256}"
download_verified "${kernel_url}" "${kernel_wheel_path}" "${KERNEL_WHEEL_SHA256}"

install_id="${FREETOKEN_VERSION}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
candidate="${VERSIONS_ROOT}/${install_id}"
trap 'rm -rf "${candidate:-}" "${ACTIVE_ROOT}.new.$$"' EXIT
"${uv_bin}" venv --python "${PYTHON_VERSION}" "${candidate}"
# FreeToken v0.1.2 pins the CUDA-compatible torch and sglang-kernel ranges itself.
# Their official PyPI wheels are CUDA 13 builds; the extra indexes are needed only
# for FlashInfer's prebuilt cubin and JIT-cache packages.
index_args=()
if [[ -n "${PACKAGE_INDEX_URL}" ]]; then
  index_args+=(--default-index "${PACKAGE_INDEX_URL}")
fi
"${uv_bin}" pip install --python "${candidate}/bin/python" \
  --index-strategy unsafe-best-match \
  "${index_args[@]}" \
  --extra-index-url https://flashinfer.ai/whl \
  --extra-index-url https://flashinfer.ai/whl/cu130 \
  "${runtime_wheel_path}[accel]" \
  "flashinfer-python[cu13]==${FLASHINFER_VERSION}" \
  "flashinfer-cubin==${FLASHINFER_VERSION}" \
  "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
  "${kernel_wheel_path}" \
  2>&1 | tee "${LOG_ROOT}/install.log"

"${candidate}/bin/ft" --help >/dev/null
if [[ ${SMOKE_TEST} -eq 1 ]]; then
  "${candidate}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch cannot access an NVIDIA GPU")
print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
PY
fi

"${uv_bin}" pip freeze --python "${candidate}/bin/python" >"${candidate}/requirements.freeze.txt"
cat >"${candidate}/install-manifest.json" <<EOF
{
  "schema_version": 1,
  "backend": "freetoken-linux-cuda",
  "freetoken_version": "${FREETOKEN_VERSION}",
  "flashinfer_version": "${FLASHINFER_VERSION}",
  "python_version": "${PYTHON_VERSION}",
  "uv_version": "${UV_VERSION}",
  "runtime_wheel_sha256": "${RUNTIME_WHEEL_SHA256}",
  "kernel_cache_wheel_sha256": "${KERNEL_WHEEL_SHA256}",
  "nvidia_driver_branch": "$(nvidia_driver_branch)",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

active_link="${ACTIVE_ROOT}.new.$$"
ln -s ".freetoken-linux-cuda-versions/${install_id}" "${active_link}"
mv -Tf "${active_link}" "${ACTIVE_ROOT}"
trap - EXIT

echo "FreeToken Linux CUDA runtime installed."
echo "  launcher: ${ACTIVE_ROOT}/bin/ft"
echo "  manifest: ${ACTIVE_ROOT}/install-manifest.json"
