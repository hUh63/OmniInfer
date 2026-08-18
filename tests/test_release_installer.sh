#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALLER="${REPO_ROOT}/scripts/install.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

make_asset() {
    local version="$1"
    local target="$2"
    local marker="$3"
    local stage="${TEST_ROOT}/stage-${version}-${target}"
    local release_dir="${TEST_ROOT}/releases/download/${version}"
    local asset="omniinfer-${version}-${target}.tar.gz"

    mkdir -p "${stage}/OmniInfer" "${release_dir}"
    printf '#!/usr/bin/env sh\nprintf "omniinfer %s\\n"\n' "${marker}" >"${stage}/OmniInfer/omniinfer"
    chmod 0755 "${stage}/OmniInfer/omniinfer"
    tar -czf "${release_dir}/${asset}" -C "${stage}" OmniInfer
}

write_checksums() {
    local version="$1"
    local release_dir="${TEST_ROOT}/releases/download/${version}"
    local asset path
    : >"${release_dir}/checksums.txt"
    for path in "${release_dir}"/omniinfer-*; do
        asset="$(basename "${path}")"
        printf '%s  %s\n' "$(sha256_file "${path}")" "${asset}" >>"${release_dir}/checksums.txt"
    done
}

make_asset v1.2.3 linux-x64 1.2.3-linux
make_asset v1.2.3 macos-arm64 1.2.3-macos
write_checksums v1.2.3
mkdir -p "${TEST_ROOT}/api/repos/example/OmniInfer/releases"
printf '{"tag_name":"v1.2.3"}\n' >"${TEST_ROOT}/api/repos/example/OmniInfer/releases/latest"

linux_bin="${TEST_ROOT}/install-linux"
bash "${INSTALLER}" \
    --repo example/OmniInfer \
    --api-url "file://${TEST_ROOT}/api" \
    --base-url "file://${TEST_ROOT}/releases/download" \
    --target linux-x64 \
    --install-dir "${linux_bin}"
[[ "$("${linux_bin}/omniinfer")" == "omniinfer 1.2.3-linux" ]] || fail "Linux fixture did not run"

# Reinstalling the same release must be safe and deterministic.
bash "${INSTALLER}" \
    --version 1.2.3 \
    --base-url "file://${TEST_ROOT}/releases/download" \
    --target linux-x64 \
    --install-dir "${linux_bin}"
[[ "$("${linux_bin}/omniinfer")" == "omniinfer 1.2.3-linux" ]] || fail "Linux reinstall changed the result"

macos_bin="${TEST_ROOT}/install-macos"
bash "${INSTALLER}" \
    --version v1.2.3 \
    --base-url "file://${TEST_ROOT}/releases/download" \
    --target macos-arm64 \
    --install-dir "${macos_bin}"
[[ "$("${macos_bin}/omniinfer")" == "omniinfer 1.2.3-macos" ]] || fail "macOS fixture did not run"

# A bad checksum must fail without overwriting the previously installed CLI.
make_asset v1.2.4 linux-x64 1.2.4-corrupt
printf '%064d  omniinfer-v1.2.4-linux-x64.tar.gz\n' 0 >"${TEST_ROOT}/releases/download/v1.2.4/checksums.txt"
if bash "${INSTALLER}" \
    --version v1.2.4 \
    --base-url "file://${TEST_ROOT}/releases/download" \
    --target linux-x64 \
    --install-dir "${linux_bin}" >/dev/null 2>&1; then
    fail "checksum mismatch was accepted"
fi
[[ "$("${linux_bin}/omniinfer")" == "omniinfer 1.2.3-linux" ]] || fail "failed install overwrote the existing CLI"

if bash "${INSTALLER}" --version '../bad' --target linux-x64 --dry-run >/dev/null 2>&1; then
    fail "invalid version was accepted"
fi
if bash "${INSTALLER}" --version v1.2.3 --target windows-x64 --dry-run >/dev/null 2>&1; then
    fail "unsupported target was accepted"
fi

echo "release installer shell tests passed"
