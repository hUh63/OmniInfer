#!/usr/bin/env bash

set -euo pipefail

BACKEND_ID="${OMNIINFER_VLA_CPP_BACKEND_ID:-vla.cpp-linux}"
BACKEND_LABEL="${OMNIINFER_VLA_CPP_BACKEND_LABEL:-vla.cpp Linux CPU}"
ENABLE_CUDA="${OMNIINFER_VLA_CPP_ENABLE_CUDA:-0}"
BUILD_TYPE="Release"
DRY_RUN=0
JOBS=""
CLEAN_BUILD=0
BOOTSTRAP_SUBMODULE=1
SMOKE_TEST=0
CHECK_DEPS=0
BUILD_FROM_SOURCE=0
USE_NATIVE=0
ENABLE_LTO=0
DEPENDENCY_PREFIX=""
CUDA_ARCHITECTURES=""

check_deps() {
  local rc=0
  _dep() {
    local cmd="$1" desc="$2" hint="$3" pkg="${4:-}"
    if command -v "${cmd}" >/dev/null 2>&1; then
      printf 'ok|%s|%s|%s|%s\n' "${cmd}" "${desc}" "${hint}" "${pkg}"
    else
      printf 'missing|%s|%s|%s|%s\n' "${cmd}" "${desc}" "${hint}" "${pkg}"
      rc=1
    fi
  }
  _dep cmake "CMake build system" "sudo apt install cmake" cmake
  _dep pkg-config "pkg-config for libzmq discovery" "sudo apt install pkg-config" pkg-config
  _dep protoc "Protocol Buffers compiler" "sudo apt install protobuf-compiler" protobuf-compiler
  return ${rc}
}

usage() {
  cat <<'EOF'
Usage: build-vla-linux.sh [options]

Options:
  --build-type <type>          CMake build type, default: Release
  --jobs <n>                   Parallel build jobs, default: nproc
  --native                     Optimize host-side kernels for the current CPU
  --portable                   Disable host-specific CPU tuning (default)
  --lto                        Enable link-time optimization
  --clean                      Remove the previous build directory before configuring
  --dependency-prefix <path>   Prefix containing protobuf/cppzmq/libzmq dependencies
  --cuda-architectures <list>  CMAKE_CUDA_ARCHITECTURES value for CUDA builds
  --no-bootstrap               Do not auto-initialize the vla.cpp git submodule
  --from-source                Build from the checked-out source submodule
  --smoke-test                 Run `vla-server --help` after the build completes
  --dry-run                    Print actions without executing them
  -h, --help                   Show this help message
EOF
}

while (($# > 0)); do
  case "$1" in
    --build-type)
      BUILD_TYPE="${2:?missing value for --build-type}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    --native)
      USE_NATIVE=1
      shift
      ;;
    --portable)
      USE_NATIVE=0
      shift
      ;;
    --lto)
      ENABLE_LTO=1
      shift
      ;;
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    --dependency-prefix)
      DEPENDENCY_PREFIX="${2:?missing value for --dependency-prefix}"
      shift 2
      ;;
    --cuda-architectures)
      CUDA_ARCHITECTURES="${2:?missing value for --cuda-architectures}"
      shift 2
      ;;
    --no-bootstrap)
      BOOTSTRAP_SUBMODULE=0
      shift
      ;;
    --from-source)
      BUILD_FROM_SOURCE=1
      shift
      ;;
    --smoke-test)
      SMOKE_TEST=1
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
    --check-deps)
      CHECK_DEPS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/${BACKEND_ID}"
VLA_ROOT="${REPO_ROOT}/framework/vla.cpp"
BUILD_ROOT="${PACKAGE_ROOT}/build/${BACKEND_ID}"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"
MODELS_ROOT="${REPO_ROOT}/.local/models"

if [[ -n "${DEPENDENCY_PREFIX}" ]]; then
  export PATH="${DEPENDENCY_PREFIX}/bin:${PATH}"
  export CMAKE_PREFIX_PATH="${DEPENDENCY_PREFIX}:${CMAKE_PREFIX_PATH:-}"
  export PKG_CONFIG_PATH="${DEPENDENCY_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

if [[ ${CHECK_DEPS} -eq 1 ]]; then
  check_deps
  exit $?
fi

if [[ ${BUILD_FROM_SOURCE} -eq 0 ]]; then
  echo "No prebuilt install path is configured for ${BACKEND_ID}." >&2
  echo "Re-run with --from-source to build from framework/vla.cpp." >&2
  exit 1
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    exit 1
  fi
}

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi
  printf '1\n'
}

ensure_vla_root() {
  if [[ -f "${VLA_ROOT}/CMakeLists.txt" ]]; then
    return
  fi

  if [[ ${BOOTSTRAP_SUBMODULE} -eq 0 ]]; then
    echo "vla.cpp source tree was not found at ${VLA_ROOT}" >&2
    echo "Run: git submodule update --init --recursive framework/vla.cpp" >&2
    exit 1
  fi

  if [[ ! -d "${REPO_ROOT}/.git" && ! -f "${REPO_ROOT}/.git" ]]; then
    echo "vla.cpp source tree was not found at ${VLA_ROOT}" >&2
    exit 1
  fi

  require_command git
  echo "vla.cpp source tree is missing. Bootstrapping the submodule..."
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  git -C ${REPO_ROOT} submodule update --init --recursive framework/vla.cpp"
    return
  fi
  git -C "${REPO_ROOT}" submodule update --init --recursive framework/vla.cpp

  if [[ ! -f "${VLA_ROOT}/CMakeLists.txt" ]]; then
    echo "Failed to prepare vla.cpp at ${VLA_ROOT}" >&2
    exit 1
  fi
}

prepare_runtime_dirs() {
  mkdir -p "${BUILD_ROOT}" "${BIN_ROOT}" "${LOG_ROOT}" "${MODELS_ROOT}"
  touch "${BIN_ROOT}/.gitkeep" "${LOG_ROOT}/.gitkeep" "${MODELS_ROOT}/.gitkeep"
}

copy_dependency_runtime_libs() {
  if [[ -z "${DEPENDENCY_PREFIX}" || ! -d "${DEPENDENCY_PREFIX}/lib" ]]; then
    return
  fi
  shopt -s nullglob
  local libs=("${DEPENDENCY_PREFIX}"/lib/*.so "${DEPENDENCY_PREFIX}"/lib/*.so.*)
  shopt -u nullglob
  if [[ ${#libs[@]} -eq 0 ]]; then
    return
  fi
  cp -a "${libs[@]}" "${BIN_ROOT}/"
}

write_vla_launcher_wrapper() {
  cat >"${BIN_ROOT}/vla-server" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}:${LD_LIBRARY_PATH:-}"
exec "${SCRIPT_DIR}/vla-server.bin" "$@"
EOF
  chmod +x "${BIN_ROOT}/vla-server"
}

install_vla_server_binary() {
  local source_binary=""
  for candidate in "${BUILD_ROOT}/vla-server" "${BUILD_ROOT}/bin/vla-server"; do
    if [[ -x "${candidate}" ]]; then
      source_binary="${candidate}"
      break
    fi
  done
  if [[ -z "${source_binary}" ]]; then
    echo "Build finished but vla-server was not found under ${BUILD_ROOT}." >&2
    exit 1
  fi
  if [[ -n "${DEPENDENCY_PREFIX}" ]]; then
    cp -a "${source_binary}" "${BIN_ROOT}/vla-server.bin"
    chmod +x "${BIN_ROOT}/vla-server.bin"
    copy_dependency_runtime_libs
    write_vla_launcher_wrapper
  else
    cp -a "${source_binary}" "${BIN_ROOT}/vla-server"
    chmod +x "${BIN_ROOT}/vla-server"
  fi
}

ensure_vla_root
require_command cmake
require_command pkg-config
require_command protoc

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

CONFIGURE_ARGS=(
  -S "${VLA_ROOT}"
  -B "${BUILD_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DBUILD_SHARED_LIBS=OFF
  -DGGML_NATIVE=$( [[ ${USE_NATIVE} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DGGML_LTO=$( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DGGML_CUDA=$( [[ ${ENABLE_CUDA} -eq 1 ]] && printf 'ON' || printf 'OFF' )
)

if [[ -n "${CUDA_ARCHITECTURES}" ]]; then
  CONFIGURE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}")
fi

if command -v ninja >/dev/null 2>&1; then
  CONFIGURE_ARGS+=(-G Ninja)
fi

echo "Configuring ${BACKEND_LABEL} build..."
echo "  cmake ${CONFIGURE_ARGS[*]}"
echo "Building vla-server..."
echo "  cmake --build ${BUILD_ROOT} --target vla-server --config ${BUILD_TYPE} -j ${JOBS}"
echo "CPU tuning mode: $( [[ ${USE_NATIVE} -eq 1 ]] && printf 'native' || printf 'portable' )"
echo "CUDA: $( [[ ${ENABLE_CUDA} -eq 1 ]] && printf 'enabled' || printf 'disabled' )"
echo "Link-time optimization: $( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'enabled' || printf 'disabled' )"
if [[ -n "${DEPENDENCY_PREFIX}" ]]; then
  echo "Dependency prefix: ${DEPENDENCY_PREFIX}"
fi
if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  echo "Cleaning previous build directory: ${BUILD_ROOT}"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi

prepare_runtime_dirs

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  rm -rf "${BUILD_ROOT}"
fi
mkdir -p "${BUILD_ROOT}"

cmake "${CONFIGURE_ARGS[@]}"
cmake --build "${BUILD_ROOT}" --target vla-server --config "${BUILD_TYPE}" -j "${JOBS}"

find "${BIN_ROOT}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +
install_vla_server_binary

if [[ ! -x "${BIN_ROOT}/vla-server" ]]; then
  echo "Build finished but vla-server launcher was not installed into ${BIN_ROOT}." >&2
  exit 1
fi

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  echo "Running smoke test..."
  "${BIN_ROOT}/vla-server" --help >/dev/null
fi

echo
echo "${BACKEND_LABEL} build complete."
echo "Binary package location: ${BIN_ROOT}"
echo "Models directory: ${MODELS_ROOT}"
echo "Next step:"
echo "  ./omniinfer backend select ${BACKEND_ID}"
echo "  ./omniinfer model load -m /absolute/path/to/vla-checkpoint.gguf"
