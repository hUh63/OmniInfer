use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::backend_installer::{InstallReporter, download_verified_asset};
use crate::prebuilt_catalog::{
    PrebuiltCatalog, PythonRuntimeEntry, PythonRuntimeVariant, RocmPackageAsset, RocmSystemRuntime,
};

const CUDA_BACKEND_ID: &str = "vllm-wsl2-cuda";
const ROCM_BACKEND_ID: &str = "vllm-wsl2-rocm";
const LAUNCHER_MANIFEST: &str = "vllm-wsl2.json";
const MANAGED_MANIFEST: &str = "managed-runtime.json";
const RUNTIME_ENV: &str = "runtime.env";
const RUNTIME_ENVIRONMENT_VERSION: u32 = 6;
const ROCM_PLATFORM_PLUGIN_VERSION: &str = "1.1.0";

const ROCM_PLATFORM_PLUGIN: &str = r#"import os

_shim_active = False
_uva_fallback_active = False


def _is_wsl():
    try:
        with open("/proc/sys/kernel/osrelease", encoding="utf-8") as release:
            return "microsoft" in release.read().lower()
    except OSError:
        return False


def _amdsmi_has_gpu():
    initialized = False
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        initialized = True
        return bool(amdsmi.amdsmi_get_processor_handles())
    except Exception:
        return False
    finally:
        if initialized:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass


def _install_amdsmi_shim(devices):
    import amdsmi

    def handles():
        return list(range(len(devices)))

    def properties(handle):
        return devices[int(handle)]

    def asic_info(handle):
        device = properties(handle)
        return {
            "asic_serial": device["asic_serial"],
            "device_id": "",
            "market_name": device["name"],
            "target_graphics_version": device["gcn_arch"],
        }

    amdsmi.amdsmi_init = lambda *args, **kwargs: None
    amdsmi.amdsmi_shut_down = lambda: None
    amdsmi.amdsmi_get_processor_handles = handles
    amdsmi.amdsmi_get_gpu_asic_info = asic_info
    amdsmi.amdsmi_get_gpu_memory_total = (
        lambda handle, memory_type: properties(handle)["total_memory"]
    )
    amdsmi.amdsmi_get_gpu_device_uuid = (
        lambda handle: properties(handle)["uuid"]
    )
    amdsmi.amdsmi_topo_get_link_type = (
        lambda handle, peer_handle: {"hops": 1, "type": 2}
    )
    amdsmi.amdsmi_topo_get_numa_node_number = lambda handle: 0


def platform_plugin():
    global _shim_active
    if _shim_active:
        return "vllm.platforms.rocm.RocmPlatform"
    if os.environ.get("HSA_ENABLE_DXG_DETECTION") != "1" or not _is_wsl():
        return None
    if _amdsmi_has_gpu():
        return None
    try:
        import torch

        if (
            torch.version.hip
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 0
        ):
            devices = []
            for index in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(index)
                uuid = str(getattr(device, "uuid", f"wsl2-rocm-{index}"))
                devices.append(
                    {
                        "asic_serial": f"0x{uuid.replace('-', '')}",
                        "gcn_arch": device.gcnArchName,
                        "name": device.name,
                        "total_memory": device.total_memory,
                        "uuid": uuid,
                    }
                )
            _install_amdsmi_shim(devices)
            _shim_active = True
            return "vllm.platforms.rocm.RocmPlatform"
    except Exception:
        pass
    return None


def general_plugin():
    global _uva_fallback_active
    if _uva_fallback_active:
        return
    if not _shim_active:
        platform_plugin()
    if not _shim_active:
        return

    from vllm.utils.platform_utils import is_uva_available

    if is_uva_available():
        return

    import torch
    from vllm.v1.worker.gpu import buffer_utils

    class WslRocmBuffer:
        def __init__(self, size, dtype):
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
            self.np = self.cpu.numpy()
            self.uva = torch.zeros(size, dtype=dtype, device="cuda")

    def copy_to_accelerator(self, value):
        self._curr = (self._curr + 1) % self.max_concurrency
        buffer = self._uva_bufs[self._curr]
        destination = buffer.cpu if isinstance(value, torch.Tensor) else buffer.np
        count = len(value)
        destination[:count] = value
        return buffer.uva[:count].copy_(buffer.cpu[:count])

    buffer_utils.UvaBuffer = WslRocmBuffer
    buffer_utils.UvaBufferPool.copy_to_uva = copy_to_accelerator
    _uva_fallback_active = True
"#;

const ROCM_PLATFORM_PLUGIN_ENTRY_POINTS: &str = r#"[vllm.platform_plugins]
omniinfer_wsl2_rocm = omniinfer_vllm_wsl2_rocm:platform_plugin

[vllm.general_plugins]
omniinfer_wsl2_rocm = omniinfer_vllm_wsl2_rocm:general_plugin
"#;

const RUNNER_SCRIPT: &str = r#"#!/bin/sh
set -eu
pid_file=$1
shift
runtime_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -f "$runtime_dir/runtime.env" ]; then
    set -a
    . "$runtime_dir/runtime.env"
    set +a
fi
mkdir -p "$(dirname "$pid_file")"
if [ -s "$pid_file" ]; then
    old_pid=$(cat "$pid_file")
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "vLLM runtime is already active with pid $old_pid" >&2
        exit 73
    fi
    rm -f "$pid_file"
fi
managed_memory_policy=1
managed_eager_policy=1
managed_chunked_prefill_policy=1
for argument in "$@"; do
    case "$argument" in
        --kv-cache-memory-bytes|--kv-cache-memory-bytes=*|--gpu-memory-utilization|--gpu-memory-utilization=*)
            managed_memory_policy=0
            ;;
        --enforce-eager|--no-enforce-eager)
            managed_eager_policy=0
            ;;
        --enable-chunked-prefill|--enable-chunked-prefill=*|--no-enable-chunked-prefill)
            managed_chunked_prefill_policy=0
            ;;
    esac
done
if [ "${HSA_ENABLE_DXG_DETECTION:-}" = "1" ]; then
    if [ "$managed_memory_policy" -eq 1 ]; then
        memory_kib=$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
        case "$memory_kib" in
            ''|*[!0-9]*) memory_kib=0 ;;
        esac
        kv_cache_bytes=$((memory_kib * 1024 / 5))
        if [ "$kv_cache_bytes" -gt 4294967296 ]; then
            kv_cache_bytes=4294967296
        fi
        if [ "$kv_cache_bytes" -ge 268435456 ]; then
            set -- "$@" --kv-cache-memory-bytes "$kv_cache_bytes"
            echo "OmniInfer: limiting WSL2 ROCm KV cache to $kv_cache_bytes bytes based on Linux memory; override with --kv-cache-memory-bytes or --gpu-memory-utilization" >&2
        fi
    fi
    if [ "$managed_eager_policy" -eq 1 ]; then
        set -- "$@" --enforce-eager
    fi
    if [ "$managed_chunked_prefill_policy" -eq 1 ]; then
        set -- "$@" --no-enable-chunked-prefill
    fi
    echo "OmniInfer: applying WSL2 ROCm compatibility defaults for eager execution and non-chunked prefill; explicit vLLM flags override each default" >&2
fi
unset argument managed_memory_policy managed_eager_policy managed_chunked_prefill_policy memory_kib kv_cache_bytes
setsid "$@" &
child=$!
printf '%s\n' "$child" > "$pid_file"
forward_signal() {
    kill -TERM "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
}
trap forward_signal HUP INT TERM
set +e
wait "$child"
status=$?
set -e
rm -f "$pid_file"
exit "$status"
"#;

const STOPPER_SCRIPT: &str = r#"#!/bin/sh
set -eu
pid_file=$1
if [ ! -s "$pid_file" ]; then
    exit 0
fi
pid=$(cat "$pid_file")
case "$pid" in
    ''|*[!0-9]*)
        echo "invalid vLLM pid file: $pid_file" >&2
        exit 74
        ;;
esac
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    exit 0
fi
kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
i=0
while [ "$i" -lt 80 ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        exit 0
    fi
    i=$((i + 1))
    sleep 0.1
done
kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
rm -f "$pid_file"
"#;

const GPU_PROBE: &str = r#"import json
import os
import torch
import vllm
from vllm.platforms import current_platform
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
expected_accelerator = os.environ["OMNIINFER_EXPECTED_ACCELERATOR"]
if expected_accelerator == "rocm" and not current_platform.is_rocm():
    raise SystemExit("vLLM did not select its ROCm platform")
if expected_accelerator == "cuda" and not current_platform.is_cuda():
    raise SystemExit("vLLM did not select its CUDA platform")
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print(json.dumps({
    "vllm_version": vllm.__version__,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "value": float(x.item()),
    "vllm_platform": type(current_platform).__module__,
}))
"#;

const NATIVE_DEPENDENCY_PROBE: &str = r#"set -eu
runtime=$1
runtime_dir=$runtime
site_packages=
for candidate in "$runtime"/venv/lib/python*/site-packages; do
    [ -d "$candidate" ] || continue
    site_packages=$candidate
    break
done
if [ -z "$site_packages" ]; then
    echo "managed site-packages directory not found" >&2
    exit 1
fi
set -a
. "$runtime/runtime.env"
set +a
if [ -n "${CC:-}" ]; then
    [ -x "$CC" ] || {
        echo "managed C compiler is not executable: $CC" >&2
        exit 1
    }
    cc_probe="$runtime/run/.cc-probe-$$.so"
    if ! printf '%s\n' 'int omniinfer_cc_probe(void) { return 0; }' |
        "$CC" -x c -shared -fPIC -o "$cc_probe" -
    then
        echo "managed C compiler probe failed: $CC" >&2
        rm -f "$cc_probe"
        exit 1
    fi
    rm -f "$cc_probe"
fi
checked=0
missing=0
for library in \
    "$site_packages"/vllm/*.so \
    "$site_packages"/flash_attn*.so \
    "$site_packages"/aiter/jit/*.so \
    "$site_packages"/xgrammar/*.so \
    "$site_packages"/torchvision/*.so \
    "$site_packages"/torchaudio/lib/*.so
do
    [ -f "$library" ] || continue
    checked=$((checked + 1))
    unresolved=$(ldd "$library" 2>&1 | grep 'not found' || true)
    if [ -n "$unresolved" ]; then
        printf '%s\n%s\n' "$library" "$unresolved" >&2
        missing=$((missing + 1))
    fi
done
if [ "$checked" -eq 0 ]; then
    echo "no managed native extensions found" >&2
    exit 1
fi
if [ "$missing" -ne 0 ]; then
    echo "$missing of $checked managed native extensions have unresolved libraries" >&2
    exit 1
fi
printf '%s\n' "$checked"
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LauncherManifest {
    schema_version: u32,
    backend: String,
    distribution: String,
    linux_launcher: String,
    linux_runner: String,
    linux_stopper: String,
    linux_pid_dir: String,
    automount_root: String,
    source: String,
    tag: String,
    python: String,
    uv_version: String,
    uv_sha256: String,
    package_version: String,
    wheel_sha256: String,
    accelerator: String,
    runtime_version: String,
    #[serde(default)]
    runtime_environment_version: u32,
}

#[derive(Debug)]
struct WslContext {
    executable: PathBuf,
    distribution: String,
    home: String,
    automount_root: String,
    install_log: PathBuf,
}

#[derive(Debug)]
struct InstallLock {
    path: PathBuf,
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub(super) fn install_wsl_python_runtime(
    backend: &str,
    runtime_dir: &Path,
    entry: &PythonRuntimeEntry,
    requested_distro: Option<&str>,
    dry_run: bool,
    reporter: &mut InstallReporter,
    catalog: &PrebuiltCatalog,
) -> Result<()> {
    if !matches!(backend, CUDA_BACKEND_ID | ROCM_BACKEND_ID) {
        anyhow::bail!("unsupported managed WSL2 backend: {backend}");
    }
    if std::env::consts::OS != "windows"
        && std::env::var_os("OMNIINFER_TEST_WSL_PLATFORM").is_none()
    {
        anyhow::bail!("{backend} is supported only by the Windows OmniInfer CLI");
    }
    let architecture = "x86_64";
    let uv = entry
        .uv
        .get(architecture)
        .ok_or_else(|| anyhow::anyhow!("{backend} has no managed uv asset for {architecture}"))?;
    let variant = entry
        .variants
        .get(architecture)
        .ok_or_else(|| anyhow::anyhow!("{backend} has no managed vLLM wheel for {architecture}"))?;
    validate_backend_accelerator(backend, variant)?;
    let torch_backend = variant
        .torch_backend
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("{backend} has no pinned PyTorch accelerator index"))?;
    let mut wsl = detect_wsl_context(requested_distro)?;
    wsl.install_log = runtime_dir.join("logs").join("install.log");
    let driver = if variant.accelerator == "cuda" {
        let driver = query_nvidia_driver()?;
        let minimum_driver = variant
            .minimum_driver
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("{backend} has no minimum NVIDIA driver"))?;
        require_minimum_driver(&driver, minimum_driver)?;
        validate_wsl_cuda_gpu(&wsl)?;
        Some(driver)
    } else {
        validate_rocm_distro(&wsl)?;
        None
    };
    let runtime_key = runtime_key(runtime_dir);
    let linux_base = format!(
        "{}/.local/share/omniinfer/runtimes/{backend}/{runtime_key}",
        wsl.home.trim_end_matches('/')
    );
    let linux_current = format!("{linux_base}/current");
    let expected = LauncherManifest {
        schema_version: 1,
        backend: backend.to_string(),
        distribution: wsl.distribution.clone(),
        linux_launcher: format!("{linux_current}/venv/bin/{}", entry.launcher),
        linux_runner: format!("{linux_current}/bin/omniinfer-vllm-run"),
        linux_stopper: format!("{linux_current}/bin/omniinfer-vllm-stop"),
        linux_pid_dir: format!("{linux_current}/run"),
        automount_root: wsl.automount_root.clone(),
        source: entry.source.clone(),
        tag: entry.tag.clone(),
        python: entry.python.clone(),
        uv_version: uv.version.clone(),
        uv_sha256: uv.sha256.clone(),
        package_version: variant.version.clone(),
        wheel_sha256: variant.sha256.clone(),
        accelerator: variant.accelerator.clone(),
        runtime_version: variant.runtime_version.clone(),
        runtime_environment_version: RUNTIME_ENVIRONMENT_VERSION,
    };

    reporter.human(format!("Managed WSL2 Python runtime: windows/{backend}"));
    reporter.human(format!("  distribution: {}", wsl.distribution));
    reporter.human(format!("  Windows runtime: {}", runtime_dir.display()));
    reporter.human(format!("  Linux runtime: {linux_current}"));
    if let Some(driver) = driver.as_deref() {
        reporter.human(format!("  NVIDIA driver: {driver}"));
    }
    reporter.human(format!(
        "  selected {} runtime: {} ({torch_backend})",
        variant.accelerator.to_ascii_uppercase(),
        variant.runtime_version,
    ));
    if let Some(system) = variant.rocm_system.as_ref() {
        reporter.human(format!(
            "  minimum AMD Software release: {}",
            system.minimum_windows_release
        ));
    }
    if variant.reported_version() != variant.version {
        reporter.human(format!(
            "  package metadata version: {}",
            variant.reported_version()
        ));
    }
    if variant.reported_runtime_version() != variant.runtime_version {
        reporter.human(format!(
            "  reported accelerator ABI: {}",
            variant.reported_runtime_version()
        ));
    }
    reporter.human(format!("  wheel sha256: {}", variant.sha256));
    reporter.event(
        "compatibility_selected",
        json!({
            "architecture": architecture,
            "distribution": wsl.distribution,
            "driver": driver,
            "minimum_driver": variant.minimum_driver,
            "accelerator": variant.accelerator,
            "runtime_version": variant.runtime_version,
            "reported_runtime_version": variant.reported_runtime_version(),
            "runtime_environment_version": RUNTIME_ENVIRONMENT_VERSION,
            "torch_backend": torch_backend,
            "wheel_url": variant.url,
            "package_version": variant.version,
            "reported_package_version": variant.reported_version(),
            "wheel_sha256": variant.sha256,
            "linux_runtime": linux_current,
        }),
    );

    if launcher_manifest_matches(runtime_dir, &expected)
        && validate_existing_system_runtime(&wsl, variant, reporter).is_ok()
        && validate_installed_runtime(
            &wsl,
            &linux_current,
            variant.reported_version(),
            &variant.accelerator,
            variant.reported_runtime_version(),
            reporter,
        )
        .is_ok()
    {
        reporter.human(format!(
            "Backend already installed and GPU-verified: {backend}"
        ));
        reporter.event(
            "already_installed",
            json!({
                "runtime_dir": runtime_dir,
                "distribution": wsl.distribution,
                "linux_runtime": linux_current,
                "launcher": runtime_dir.join("bin").join(LAUNCHER_MANIFEST),
            }),
        );
        return Ok(());
    }

    if dry_run {
        reporter.event(
            "asset_planned",
            json!({
                "role": "uv",
                "url": uv.url,
                "expected_sha256": uv.sha256,
            }),
        );
        if let Some(system) = variant.rocm_system.as_ref() {
            reporter.event(
                "asset_planned",
                json!({
                    "role": "ROCm repository key",
                    "url": system.repository_key.url,
                    "expected_sha256": system.repository_key.sha256,
                }),
            );
            reporter.event(
                "asset_planned",
                json!({
                    "role": "ROCDXG runtime",
                    "url": system.rocdxg.url,
                    "expected_sha256": system.rocdxg.sha256,
                }),
            );
        }
        reporter.event(
            "dry_run_completed",
            json!({
                "runtime_dir": runtime_dir,
                "distribution": wsl.distribution,
                "linux_runtime": linux_current,
                "wheel_url": variant.url,
                "package_version": variant.version,
                "reported_package_version": variant.reported_version(),
                "reported_runtime_version": variant.reported_runtime_version(),
                "wheel_sha256": variant.sha256,
            }),
        );
        return Ok(());
    }

    fs::create_dir_all(runtime_dir)
        .with_context(|| format!("create runtime directory {}", runtime_dir.display()))?;
    let _lock = acquire_install_lock(runtime_dir)?;
    ensure_runtime_not_active(&wsl, &linux_current)?;
    let uv_bytes = download_verified_asset(catalog, &uv.url, &uv.sha256, "uv", reporter)?;
    let local_uv = extract_uv(runtime_dir, &uv_bytes)?;
    let wsl_uv_source = wsl_path(&wsl, &local_uv)?;
    let suffix = unique_suffix();
    let linux_staging = format!("{linux_base}/installing-{suffix}");
    let linux_backup = format!("{linux_base}/backup-{suffix}");
    let linux_uv = format!("{linux_base}/tools/uv-{}", uv.version);

    reporter.event(
        "staging_started",
        json!({
            "distribution": wsl.distribution,
            "staging": linux_staging,
        }),
    );
    run_wsl_checked(
        &wsl,
        [
            "mkdir",
            "-p",
            linux_base.as_str(),
            format!("{linux_base}/tools").as_str(),
        ],
        reporter,
        "prepare WSL runtime directories",
    )?;
    if let Some(system) = variant.rocm_system.as_ref() {
        ensure_rocm_system_runtime(&wsl, system, runtime_dir, catalog, reporter)?;
        validate_wsl_rocm_gpu(&wsl, system)?;
    }
    run_wsl_checked(
        &wsl,
        ["cp", wsl_uv_source.as_str(), linux_uv.as_str()],
        reporter,
        "stage managed uv",
    )?;
    run_wsl_checked(
        &wsl,
        ["chmod", "0755", linux_uv.as_str()],
        reporter,
        "mark managed uv executable",
    )?;
    run_wsl_checked(
        &wsl,
        ["rm", "-rf", linux_staging.as_str()],
        reporter,
        "clear stale WSL staging runtime",
    )?;
    run_wsl_checked(
        &wsl,
        [
            "mkdir",
            "-p",
            format!("{linux_staging}/bin").as_str(),
            format!("{linux_staging}/run").as_str(),
        ],
        reporter,
        "create WSL staging runtime",
    )?;

    let install_result = (|| {
        run_wsl_checked(
            &wsl,
            [
                linux_uv.as_str(),
                "venv",
                "--no-project",
                "--relocatable",
                "--python",
                entry.python.as_str(),
                format!("{linux_staging}/venv").as_str(),
            ],
            reporter,
            "create managed WSL Python environment",
        )?;
        let requirement = format!("{}#sha256={}", variant.url, variant.sha256);
        let python = format!("{linux_staging}/venv/bin/python");
        let mut install_args = vec![
            linux_uv.clone(),
            "pip".to_string(),
            "install".to_string(),
            "--python".to_string(),
            python,
        ];
        if variant.accelerator == "cuda" {
            install_args.extend([
                "--torch-backend".to_string(),
                torch_backend.to_string(),
                "--index-strategy".to_string(),
                "first-index".to_string(),
            ]);
        } else {
            let index_url = variant
                .index_url
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("{backend} has no pinned ROCm wheel index"))?;
            install_args.extend([
                "--extra-index-url".to_string(),
                index_url.to_string(),
                "--index-strategy".to_string(),
                "unsafe-best-match".to_string(),
            ]);
        }
        install_args.push(requirement);
        run_wsl_checked(
            &wsl,
            install_args.iter().map(String::as_str),
            reporter,
            "install pinned vLLM wheel",
        )?;
        run_wsl_checked(
            &wsl,
            [
                linux_uv.as_str(),
                "pip",
                "check",
                "--python",
                format!("{linux_staging}/venv/bin/python").as_str(),
            ],
            reporter,
            "check managed WSL Python dependencies",
        )?;
        write_wsl_file(
            &wsl,
            &format!("{linux_staging}/bin/omniinfer-vllm-run"),
            RUNNER_SCRIPT.as_bytes(),
            true,
        )?;
        write_wsl_file(
            &wsl,
            &format!("{linux_staging}/bin/omniinfer-vllm-stop"),
            STOPPER_SCRIPT.as_bytes(),
            true,
        )?;
        write_wsl_file(
            &wsl,
            &format!("{linux_staging}/{RUNTIME_ENV}"),
            runtime_environment(&variant.accelerator).as_bytes(),
            false,
        )?;
        if variant.accelerator == "rocm" {
            write_rocm_platform_plugin(&wsl, &linux_staging)?;
        }
        let managed_manifest = json!({
            "schema_version": 1,
            "backend": backend,
            "source": entry.source,
            "tag": entry.tag,
            "python": entry.python,
            "uv_version": uv.version,
            "uv_sha256": uv.sha256,
            "wheel_url": variant.url,
            "package_version": variant.version,
            "reported_package_version": variant.reported_version(),
            "wheel_sha256": variant.sha256,
            "accelerator": variant.accelerator,
            "runtime_version": variant.runtime_version,
            "reported_runtime_version": variant.reported_runtime_version(),
            "runtime_environment_version": RUNTIME_ENVIRONMENT_VERSION,
            "rocm_platform_plugin_version": if variant.accelerator == "rocm" {
                Some(ROCM_PLATFORM_PLUGIN_VERSION)
            } else {
                None
            },
            "torch_backend": torch_backend,
            "minimum_driver": variant.minimum_driver,
            "driver": driver,
            "build_commit": variant.build_commit,
            "index_url": variant.index_url,
            "rocm_system": variant.rocm_system,
        });
        write_wsl_file(
            &wsl,
            &format!("{linux_staging}/{MANAGED_MANIFEST}"),
            serde_json::to_string_pretty(&managed_manifest)?.as_bytes(),
            false,
        )?;
        validate_runtime_path(
            &wsl,
            &linux_staging,
            variant.reported_version(),
            &variant.accelerator,
            variant.reported_runtime_version(),
            reporter,
        )?;
        activate_runtime(
            &wsl,
            &linux_base,
            &linux_staging,
            &linux_current,
            &linux_backup,
            reporter,
        )?;
        if let Err(error) = validate_runtime_path(
            &wsl,
            &linux_current,
            variant.reported_version(),
            &variant.accelerator,
            variant.reported_runtime_version(),
            reporter,
        ) {
            rollback_runtime(&wsl, &linux_current, &linux_backup, reporter)?;
            return Err(error.context("post-activation WSL runtime validation failed"));
        }
        Ok::<(), anyhow::Error>(())
    })();
    if let Err(error) = install_result {
        let _ = run_wsl(&wsl, ["rm", "-rf", linux_staging.as_str()], None);
        return Err(error);
    }

    write_launcher_manifest(runtime_dir, &expected)?;
    let _ = run_wsl(&wsl, ["rm", "-rf", linux_backup.as_str()], None);
    reporter.human(format!(
        "Managed WSL2 backend installed and GPU-verified: {}",
        runtime_dir.join("bin").join(LAUNCHER_MANIFEST).display()
    ));
    reporter.event(
        "completed",
        json!({
            "runtime_dir": runtime_dir,
            "distribution": wsl.distribution,
            "linux_runtime": linux_current,
            "launcher": runtime_dir.join("bin").join(LAUNCHER_MANIFEST),
            "manifest": format!("{linux_current}/{MANAGED_MANIFEST}"),
        }),
    );
    Ok(())
}

fn detect_wsl_context(requested_distro: Option<&str>) -> Result<WslContext> {
    let executable = std::env::var_os("OMNIINFER_WSL_EXE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("wsl.exe"));
    let quiet = run_command(&executable, ["--list", "--quiet"], None)
        .with_context(|| format!("run {} --list --quiet", executable.display()))?;
    if !quiet.status.success() {
        anyhow::bail!(
            "WSL is unavailable. Enable WSL2 and install Ubuntu before installing a managed vLLM backend: {}",
            decode_output(&quiet.stderr).trim()
        );
    }
    let names = decode_output(&quiet.stdout)
        .lines()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let verbose = run_command(&executable, ["--list", "--verbose"], None)?;
    let verbose_text = decode_output(&verbose.stdout);
    let eligible = eligible_distro_names(&names, &verbose_text);
    let distribution = if let Some(requested) = requested_distro
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if !names
            .iter()
            .any(|name| name.eq_ignore_ascii_case(requested))
        {
            anyhow::bail!(
                "WSL distribution {requested:?} is not installed; available distributions: {}",
                names.join(", ")
            );
        }
        if is_internal_distro(requested) {
            anyhow::bail!(
                "WSL distribution {requested:?} is reserved for another application and cannot host OmniInfer"
            );
        }
        if !distro_is_wsl2(requested, &verbose_text) {
            anyhow::bail!("WSL distribution {requested:?} is not running as WSL2");
        }
        names
            .iter()
            .find(|name| name.eq_ignore_ascii_case(requested))
            .cloned()
            .expect("requested distribution was checked")
    } else if let Some(default) =
        default_distro(&verbose_text).filter(|name| eligible.iter().any(|item| item == name))
    {
        default
    } else if eligible.len() == 1 {
        eligible[0].clone()
    } else if eligible.is_empty() {
        anyhow::bail!(
            "no user WSL2 distribution is available. Install Ubuntu with `wsl --install -d Ubuntu`, reboot if requested, then rerun the command"
        );
    } else {
        anyhow::bail!(
            "multiple WSL2 distributions are available ({}); select one with --wsl-distro",
            eligible.join(", ")
        );
    };
    let home = run_wsl_text(
        &executable,
        &distribution,
        ["sh", "-c", "printf %s \"$HOME\""],
    )?;
    if !home.starts_with('/') {
        anyhow::bail!("WSL distribution {distribution:?} returned an invalid HOME path");
    }
    let c_root = run_wsl_text(&executable, &distribution, ["wslpath", "-a", "-u", r"C:\"])?;
    let automount_root = automount_root_from_c_path(&c_root)
        .ok_or_else(|| anyhow::anyhow!("WSL distribution returned an invalid C: automount path"))?;
    let arch = run_wsl_text(&executable, &distribution, ["uname", "-m"])?;
    if arch.trim() != "x86_64" {
        anyhow::bail!(
            "managed vLLM requires an x86_64 WSL2 distribution, found {}",
            arch.trim()
        );
    }
    Ok(WslContext {
        executable,
        distribution,
        home: home.trim().to_string(),
        automount_root,
        install_log: PathBuf::new(),
    })
}

fn validate_wsl_cuda_gpu(wsl: &WslContext) -> Result<()> {
    let output = run_wsl(
        wsl,
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        None,
    )
    .context("query NVIDIA GPU from WSL2")?;
    if !output.status.success() || output.stdout.is_empty() {
        anyhow::bail!(
            "NVIDIA CUDA is not available inside WSL2 distribution {:?}: {}",
            wsl.distribution,
            decode_output(&output.stderr).trim()
        );
    }
    Ok(())
}

fn validate_backend_accelerator(backend: &str, variant: &PythonRuntimeVariant) -> Result<()> {
    let expected = match backend {
        CUDA_BACKEND_ID => "cuda",
        ROCM_BACKEND_ID => "rocm",
        _ => anyhow::bail!("unsupported managed WSL2 backend: {backend}"),
    };
    if variant.accelerator != expected {
        anyhow::bail!(
            "{backend} catalog accelerator mismatch: expected {expected}, found {}",
            variant.accelerator
        );
    }
    Ok(())
}

fn validate_rocm_distro(wsl: &WslContext) -> Result<()> {
    let release = run_wsl_text(
        &wsl.executable,
        &wsl.distribution,
        [
            "sh",
            "-c",
            ". /etc/os-release; printf '%s %s' \"$ID\" \"$VERSION_ID\"",
        ],
    )
    .context("query WSL2 Linux release for ROCm")?;
    if release.trim() != "ubuntu 24.04" {
        anyhow::bail!(
            "{ROCM_BACKEND_ID} requires an Ubuntu 24.04 WSL2 distribution because its ROCm packages are pinned to noble; found {release:?}"
        );
    }
    Ok(())
}

fn ensure_rocm_system_runtime(
    wsl: &WslContext,
    system: &RocmSystemRuntime,
    runtime_dir: &Path,
    catalog: &PrebuiltCatalog,
    reporter: &mut InstallReporter,
) -> Result<()> {
    if let Ok(installed) = validate_rocm_system_versions(wsl, system)
        && validate_wsl_rocm_gpu(wsl, system).is_ok()
    {
        reporter.event(
            "system_runtime_verified",
            json!({
                "accelerator": "rocm",
                "reused": true,
                "packages": installed.lines().collect::<Vec<_>>(),
            }),
        );
        return Ok(());
    }
    let repository_key = download_verified_asset(
        catalog,
        &system.repository_key.url,
        &system.repository_key.sha256,
        "ROCm repository key",
        reporter,
    )?;
    let rocdxg = download_verified_asset(
        catalog,
        &system.rocdxg.url,
        &system.rocdxg.sha256,
        "ROCDXG runtime",
        reporter,
    )?;
    let key_source = format!(
        "/var/cache/omniinfer/rocm-{}.asc",
        system.repository_key.sha256
    );
    let rocdxg_source = format!("/var/cache/omniinfer/rocdxg-{}.deb", system.rocdxg.sha256);
    let key_output = format!("of={key_source}");
    let rocdxg_output = format!("of={rocdxg_source}");

    run_wsl_as_checked(
        wsl,
        Some("root"),
        [
            "install",
            "-d",
            "-m",
            "0755",
            "/etc/apt/keyrings",
            "/var/cache/omniinfer",
        ],
        None,
        reporter,
        "prepare protected ROCm system directories",
    )?;
    run_wsl_as_checked(
        wsl,
        Some("root"),
        ["dd", key_output.as_str(), "status=none"],
        Some(&repository_key),
        reporter,
        "stage verified ROCm repository key",
    )?;
    run_wsl_as_checked(
        wsl,
        Some("root"),
        ["dd", rocdxg_output.as_str(), "status=none"],
        Some(&rocdxg),
        reporter,
        "stage verified ROCDXG runtime",
    )?;
    verify_wsl_sha256(
        wsl,
        &key_source,
        &system.repository_key.sha256,
        "ROCm repository key",
    )?;
    verify_wsl_sha256(wsl, &rocdxg_source, &system.rocdxg.sha256, "ROCDXG runtime")?;
    run_wsl_as_checked(
        wsl,
        Some("root"),
        [
            "install",
            "-m",
            "0644",
            key_source.as_str(),
            "/etc/apt/keyrings/omniinfer-rocm.asc",
        ],
        None,
        reporter,
        "install verified ROCm repository key",
    )?;
    let apt_source = format!(
        "deb [arch=amd64 signed-by=/etc/apt/keyrings/omniinfer-rocm.asc] {}\n",
        system.apt_repository
    );
    run_wsl_as_checked(
        wsl,
        Some("root"),
        ["tee", "/etc/apt/sources.list.d/omniinfer-rocm.list"],
        Some(apt_source.as_bytes()),
        reporter,
        "configure pinned ROCm apt repository",
    )?;
    run_wsl_as_checked(
        wsl,
        Some("root"),
        ["apt-get", "update"],
        None,
        reporter,
        "refresh ROCm package metadata",
    )?;
    let package_cache = prepare_rocm_apt_cache(wsl, system, runtime_dir, reporter)?;
    let mut install_args = vec![
        "env".to_string(),
        "DEBIAN_FRONTEND=noninteractive".to_string(),
        "apt-get".to_string(),
        "install".to_string(),
        "--no-install-recommends".to_string(),
        "--allow-downgrades".to_string(),
        "-y".to_string(),
    ];
    install_args.extend(
        system
            .packages
            .iter()
            .map(|(name, version)| format!("{name}={version}")),
    );
    run_wsl_as_checked(
        wsl,
        Some("root"),
        install_args.iter().map(String::as_str),
        None,
        reporter,
        "install pinned ROCm runtime packages",
    )?;
    run_wsl_as_checked(
        wsl,
        Some("root"),
        ["dpkg", "-i", rocdxg_source.as_str()],
        None,
        reporter,
        "install verified ROCDXG runtime",
    )?;
    run_wsl_as_checked(
        wsl,
        Some("root"),
        ["/sbin/ldconfig"],
        None,
        reporter,
        "refresh ROCm runtime linker cache",
    )?;
    let installed = validate_rocm_system_versions(wsl, system)?;
    if package_cache.exists() {
        fs::remove_dir_all(&package_cache).with_context(|| {
            format!(
                "remove verified Windows ROCm package cache {}",
                package_cache.display()
            )
        })?;
    }
    reporter.event(
        "system_runtime_verified",
        json!({
            "accelerator": "rocm",
            "packages": installed.lines().collect::<Vec<_>>(),
        }),
    );
    Ok(())
}

#[derive(Debug, Clone)]
struct RocmPackageDownload {
    name: String,
    asset: RocmPackageAsset,
    verified_path: PathBuf,
    partial_path: PathBuf,
    asset_index: usize,
    asset_count: usize,
}

fn prepare_rocm_apt_cache(
    wsl: &WslContext,
    system: &RocmSystemRuntime,
    runtime_dir: &Path,
    reporter: &mut InstallReporter,
) -> Result<PathBuf> {
    let installed =
        query_installed_package_versions(wsl, system.package_assets.keys().map(String::as_str))?;
    let cache_dir = runtime_dir
        .join("downloads")
        .join(format!("rocm-{}", system.repository_key.version));
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("create ROCm download cache {}", cache_dir.display()))?;

    let mut downloads = Vec::new();
    let asset_count = system.package_assets.len();
    for (asset_index, (name, asset)) in system.package_assets.iter().enumerate() {
        if installed
            .get(name)
            .is_some_and(|version| version == &asset.version)
        {
            reporter.event(
                "package_download_skipped",
                json!({
                    "role": "ROCm package",
                    "package": name,
                    "version": asset.version,
                    "reason": "exact_version_installed",
                    "asset_index": asset_index + 1,
                    "asset_count": asset_count,
                }),
            );
            continue;
        }
        let apt_cache_path = format!("/var/cache/apt/archives/{}", asset.filename);
        if wsl_file_matches_sha256(wsl, &apt_cache_path, &asset.sha256)? {
            reporter.event(
                "checksum_verified",
                json!({
                    "role": format!("ROCm package {name}"),
                    "package": name,
                    "version": asset.version,
                    "url": asset.url,
                    "bytes": asset.size,
                    "sha256": asset.sha256,
                    "expected_sha256": asset.sha256,
                    "source": "wsl_apt_cache",
                    "asset_index": asset_index + 1,
                    "asset_count": asset_count,
                }),
            );
            continue;
        }
        let verified_path = cache_dir.join(format!("{}.deb", asset.sha256));
        let partial_path = cache_dir.join(format!("{}.partial", asset.sha256));
        downloads.push(RocmPackageDownload {
            name: name.clone(),
            asset: asset.clone(),
            verified_path,
            partial_path,
            asset_index: asset_index + 1,
            asset_count,
        });
    }

    download_rocm_packages(&downloads, reporter)?;
    if !downloads.is_empty() {
        run_wsl_as_checked(
            wsl,
            Some("root"),
            [
                "install",
                "-d",
                "-m",
                "0755",
                "/var/cache/omniinfer/rocm-packages",
            ],
            None,
            reporter,
            "prepare protected ROCm package cache",
        )?;
    }
    for download in &downloads {
        let staged = format!(
            "/var/cache/omniinfer/rocm-packages/{}.deb",
            download.asset.sha256
        );
        let output_arg = format!("of={staged}");
        run_wsl_as_file_checked(
            wsl,
            Some("root"),
            ["dd", output_arg.as_str(), "status=none"],
            &download.verified_path,
            reporter,
            &format!("stage verified ROCm package {}", download.name),
        )?;
        verify_wsl_sha256(
            wsl,
            &staged,
            &download.asset.sha256,
            &format!("ROCm package {}", download.name),
        )?;
        let apt_cache_path = format!("/var/cache/apt/archives/{}", download.asset.filename);
        run_wsl_as_checked(
            wsl,
            Some("root"),
            [
                "install",
                "-m",
                "0644",
                staged.as_str(),
                apt_cache_path.as_str(),
            ],
            None,
            reporter,
            &format!("populate APT cache for ROCm package {}", download.name),
        )?;
        verify_wsl_sha256(
            wsl,
            &apt_cache_path,
            &download.asset.sha256,
            &format!("APT cache package {}", download.name),
        )?;
        let _ = run_wsl_as(wsl, Some("root"), ["rm", "-f", staged.as_str()], None);
        reporter.event(
            "package_cache_populated",
            json!({
                "role": "ROCm package",
                "package": download.name,
                "version": download.asset.version,
                "filename": download.asset.filename,
                "sha256": download.asset.sha256,
                "asset_index": download.asset_index,
                "asset_count": download.asset_count,
            }),
        );
    }
    Ok(cache_dir)
}

fn query_installed_package_versions<'a>(
    wsl: &WslContext,
    packages: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeMap<String, String>> {
    let mut args = vec![
        "dpkg-query".to_string(),
        "-W".to_string(),
        "-f=${Package}=${Version}\n".to_string(),
    ];
    args.extend(packages.into_iter().map(str::to_string));
    let output = run_wsl(wsl, args.iter().map(String::as_str), None)
        .context("query installed ROCm package versions")?;
    if !output.status.success() && output.status.code() != Some(1) {
        require_success(&output, "query installed ROCm package versions")?;
    }
    let mut versions = BTreeMap::new();
    for line in decode_output(&output.stdout).lines() {
        if let Some((name, version)) = line.trim().split_once('=') {
            versions.insert(name.to_string(), version.to_string());
        }
    }
    Ok(versions)
}

fn wsl_file_matches_sha256(wsl: &WslContext, path: &str, expected: &str) -> Result<bool> {
    let output = run_wsl_as(wsl, Some("root"), ["sha256sum", path], None)
        .with_context(|| format!("inspect WSL2 package cache {path}"))?;
    if !output.status.success() {
        return Ok(false);
    }
    Ok(decode_output(&output.stdout)
        .split_whitespace()
        .next()
        .is_some_and(|actual| actual.eq_ignore_ascii_case(expected)))
}

fn download_rocm_packages(
    downloads: &[RocmPackageDownload],
    reporter: &mut InstallReporter,
) -> Result<()> {
    let mut pending_https = Vec::new();
    for download in downloads {
        if verified_file_matches(download)? {
            emit_rocm_package_checksum(download, true, reporter);
            continue;
        }
        if promote_complete_partial(download)? {
            emit_rocm_package_checksum(download, true, reporter);
            continue;
        }
        if let Some(path) = download.asset.url.strip_prefix("file://") {
            reporter.event(
                "download_started",
                json!({
                    "role": format!("ROCm package {}", download.name),
                    "package": download.name,
                    "asset_index": download.asset_index,
                    "asset_count": download.asset_count,
                    "url": download.asset.url,
                }),
            );
            fs::copy(path, &download.partial_path).with_context(|| {
                format!(
                    "copy ROCm package fixture {} to {}",
                    path,
                    download.partial_path.display()
                )
            })?;
            reporter.event(
                "download_progress",
                json!({
                    "role": format!("ROCm package {}", download.name),
                    "package": download.name,
                    "asset_index": download.asset_index,
                    "asset_count": download.asset_count,
                    "url": download.asset.url,
                    "bytes_downloaded": fs::metadata(&download.partial_path)?.len(),
                    "bytes_total": download.asset.size,
                }),
            );
        } else {
            pending_https.push(download);
        }
    }

    if !pending_https.is_empty() {
        download_rocm_packages_with_curl(&pending_https, reporter)?;
    }
    for download in downloads {
        if download.verified_path.exists() {
            continue;
        }
        let actual_size = fs::metadata(&download.partial_path)
            .with_context(|| {
                format!(
                    "downloaded ROCm package {} is missing from {}",
                    download.name,
                    download.partial_path.display()
                )
            })?
            .len();
        let actual_sha256 = sha256_file(&download.partial_path)?;
        if actual_size != download.asset.size
            || !actual_sha256.eq_ignore_ascii_case(&download.asset.sha256)
        {
            reporter.event(
                "checksum_failed",
                json!({
                    "role": format!("ROCm package {}", download.name),
                    "package": download.name,
                    "asset_index": download.asset_index,
                    "asset_count": download.asset_count,
                    "url": download.asset.url,
                    "expected_bytes": download.asset.size,
                    "actual_bytes": actual_size,
                    "expected_sha256": download.asset.sha256,
                    "actual_sha256": actual_sha256,
                }),
            );
            anyhow::bail!(
                "ROCm package {} checksum mismatch: expected {} bytes / {}, got {} bytes / {}",
                download.name,
                download.asset.size,
                download.asset.sha256,
                actual_size,
                actual_sha256
            );
        }
        fs::rename(&download.partial_path, &download.verified_path).with_context(|| {
            format!(
                "commit verified ROCm package {} to {}",
                download.name,
                download.verified_path.display()
            )
        })?;
        emit_rocm_package_checksum(download, false, reporter);
    }
    Ok(())
}

fn download_rocm_packages_with_curl(
    pending: &[&RocmPackageDownload],
    reporter: &mut InstallReporter,
) -> Result<()> {
    let cache_dir = pending[0]
        .partial_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("ROCm package cache has no parent directory"))?;
    let stderr_path = cache_dir.join("curl.stderr.log");
    let stderr_file = File::create(&stderr_path)
        .with_context(|| format!("create curl log {}", stderr_path.display()))?;
    let mut command = Command::new("curl.exe");
    command.args([
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "60",
        "--parallel",
        "--parallel-max",
        "4",
        "--silent",
        "--show-error",
    ]);
    for download in pending {
        reporter.event(
            "download_started",
            json!({
                "role": format!("ROCm package {}", download.name),
                "package": download.name,
                "asset_index": download.asset_index,
                "asset_count": download.asset_count,
                "url": download.asset.url,
                "resumable": true,
            }),
        );
        command
            .arg("--continue-at")
            .arg("-")
            .arg("--output")
            .arg(&download.partial_path)
            .arg(&download.asset.url);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file));
    hide_child_window(&mut command);
    let mut child = command
        .spawn()
        .context("start Windows curl.exe for resumable ROCm package downloads")?;
    let mut last_reported = BTreeMap::<String, u64>::new();
    let status = loop {
        if let Some(status) = child.try_wait().context("poll Windows curl.exe")? {
            break status;
        }
        for download in pending {
            let downloaded = fs::metadata(&download.partial_path)
                .map(|metadata| metadata.len())
                .unwrap_or_default();
            let previous = last_reported
                .get(&download.name)
                .copied()
                .unwrap_or_default();
            if downloaded >= previous.saturating_add(8 * 1024 * 1024)
                || downloaded >= download.asset.size
            {
                reporter.event(
                    "download_progress",
                    json!({
                        "role": format!("ROCm package {}", download.name),
                        "package": download.name,
                        "asset_index": download.asset_index,
                        "asset_count": download.asset_count,
                        "url": download.asset.url,
                        "bytes_downloaded": downloaded,
                        "bytes_total": download.asset.size,
                    }),
                );
                last_reported.insert(download.name.clone(), downloaded);
            }
        }
        thread::sleep(Duration::from_secs(1));
    };
    for download in pending {
        let downloaded = fs::metadata(&download.partial_path)
            .map(|metadata| metadata.len())
            .unwrap_or_default();
        reporter.event(
            "download_progress",
            json!({
                "role": format!("ROCm package {}", download.name),
                "package": download.name,
                "asset_index": download.asset_index,
                "asset_count": download.asset_count,
                "url": download.asset.url,
                "bytes_downloaded": downloaded,
                "bytes_total": download.asset.size,
            }),
        );
    }
    if !status.success() {
        let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
        anyhow::bail!(
            "Windows curl.exe failed while downloading ROCm packages with {status}: {}. Re-run the install to resume verified partial downloads.",
            stderr.trim()
        );
    }
    Ok(())
}

fn verified_file_matches(download: &RocmPackageDownload) -> Result<bool> {
    if !download.verified_path.exists() {
        return Ok(false);
    }
    let size = fs::metadata(&download.verified_path)?.len();
    if size == download.asset.size
        && sha256_file(&download.verified_path)?.eq_ignore_ascii_case(&download.asset.sha256)
    {
        return Ok(true);
    }
    fs::remove_file(&download.verified_path).with_context(|| {
        format!(
            "remove invalid cached ROCm package {}",
            download.verified_path.display()
        )
    })?;
    Ok(false)
}

fn promote_complete_partial(download: &RocmPackageDownload) -> Result<bool> {
    if !download.partial_path.exists() {
        return Ok(false);
    }
    let size = fs::metadata(&download.partial_path)?.len();
    if size < download.asset.size {
        return Ok(false);
    }
    if size == download.asset.size
        && sha256_file(&download.partial_path)?.eq_ignore_ascii_case(&download.asset.sha256)
    {
        fs::rename(&download.partial_path, &download.verified_path).with_context(|| {
            format!(
                "promote completed ROCm package {} to {}",
                download.name,
                download.verified_path.display()
            )
        })?;
        return Ok(true);
    }
    fs::remove_file(&download.partial_path).with_context(|| {
        format!(
            "remove invalid partial ROCm package {}",
            download.partial_path.display()
        )
    })?;
    Ok(false)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("open {} for SHA256 verification", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("read {} for SHA256 verification", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn emit_rocm_package_checksum(
    download: &RocmPackageDownload,
    cached: bool,
    reporter: &mut InstallReporter,
) {
    reporter.event(
        "checksum_verified",
        json!({
            "role": format!("ROCm package {}", download.name),
            "package": download.name,
            "version": download.asset.version,
            "asset_index": download.asset_index,
            "asset_count": download.asset_count,
            "url": download.asset.url,
            "bytes": download.asset.size,
            "sha256": download.asset.sha256,
            "expected_sha256": download.asset.sha256,
            "source": if cached { "windows_cache" } else { "download" },
        }),
    );
}

fn validate_existing_system_runtime(
    wsl: &WslContext,
    variant: &PythonRuntimeVariant,
    reporter: &mut InstallReporter,
) -> Result<()> {
    let Some(system) = variant.rocm_system.as_ref() else {
        return Ok(());
    };
    let installed = validate_rocm_system_versions(wsl, system)?;
    validate_wsl_rocm_gpu(wsl, system)?;
    reporter.event(
        "system_runtime_verified",
        json!({
            "accelerator": "rocm",
            "packages": installed.lines().collect::<Vec<_>>(),
        }),
    );
    Ok(())
}

fn validate_rocm_system_versions(wsl: &WslContext, system: &RocmSystemRuntime) -> Result<String> {
    let package_requirements = system
        .packages
        .iter()
        .map(|(name, version)| format!("{name}={version}"))
        .collect::<Vec<_>>();
    let mut query_args = vec![
        "dpkg-query".to_string(),
        "-W".to_string(),
        "-f=${Package}=${Version}\n".to_string(),
    ];
    query_args.extend(system.packages.keys().cloned());
    query_args.push("rocdxg-roct".to_string());
    let versions = run_wsl(wsl, query_args.iter().map(String::as_str), None);
    let versions = versions.context("verify installed ROCm system package versions")?;
    require_success(&versions, "verify installed ROCm system package versions")?;
    let installed = decode_output(&versions.stdout);
    let rocdxg_requirement = format!("rocdxg-roct={}", system.rocdxg.version);
    for expected in package_requirements
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(rocdxg_requirement.as_str()))
    {
        if !installed.lines().any(|line| line.trim() == expected) {
            anyhow::bail!(
                "installed ROCm system runtime does not match the pinned catalog: missing {expected}"
            );
        }
    }
    Ok(installed)
}

fn verify_wsl_sha256(wsl: &WslContext, path: &str, expected: &str, role: &str) -> Result<()> {
    let output = run_wsl_as(wsl, Some("root"), ["sha256sum", path], None)
        .with_context(|| format!("verify staged {role} in WSL2"))?;
    require_success(&output, &format!("verify staged {role} in WSL2"))?;
    let actual = decode_output(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if actual != expected.to_ascii_lowercase() {
        anyhow::bail!(
            "staged {role} checksum mismatch inside WSL2: expected {expected}, got {actual}"
        );
    }
    Ok(())
}

fn validate_wsl_rocm_gpu(wsl: &WslContext, system: &RocmSystemRuntime) -> Result<()> {
    let output = run_wsl(
        wsl,
        [
            "env",
            "HSA_ENABLE_DXG_DETECTION=1",
            "/opt/rocm/bin/rocminfo",
        ],
        None,
    )
    .context("query AMD GPU through ROCDXG in WSL2")?;
    if !output.status.success() {
        anyhow::bail!(
            "AMD ROCm is not available inside WSL2 distribution {:?}; install AMD Software {} or newer and retry: {}",
            wsl.distribution,
            system.minimum_windows_release,
            decode_output(&output.stderr).trim()
        );
    }
    let text = decode_output(&output.stdout);
    if !system.required_gfx.iter().any(|gfx| text.contains(gfx)) {
        anyhow::bail!(
            "ROCm detected no supported Ryzen GPU target (expected one of {}); install AMD Software {} or newer and verify Ryzen WSL support",
            system.required_gfx.join(", "),
            system.minimum_windows_release
        );
    }
    Ok(())
}

fn runtime_environment(accelerator: &str) -> String {
    let mut environment = r#"omniinfer_managed_library_path=
for omniinfer_library_dir in \
    "$runtime_dir"/venv/lib/python*/site-packages/torch/lib \
    "$runtime_dir"/venv/lib/python*/site-packages/tvm_ffi/lib \
    "$runtime_dir"/venv/lib/python*/site-packages/torchaudio/lib \
    "$runtime_dir"/venv/lib/python*/site-packages/*.libs \
    "$runtime_dir"/venv/lib/python*/site-packages/nvidia/*/lib
do
    [ -d "$omniinfer_library_dir" ] || continue
    if [ -n "$omniinfer_managed_library_path" ]; then
        omniinfer_managed_library_path="$omniinfer_managed_library_path:$omniinfer_library_dir"
    else
        omniinfer_managed_library_path=$omniinfer_library_dir
    fi
done
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$omniinfer_managed_library_path:$LD_LIBRARY_PATH"
else
    LD_LIBRARY_PATH=$omniinfer_managed_library_path
fi
export LD_LIBRARY_PATH
unset omniinfer_library_dir omniinfer_managed_library_path
"#
    .to_string();
    if accelerator == "rocm" {
        environment.push_str(
            r#"HSA_ENABLE_DXG_DETECTION=1
CC=/opt/rocm/llvm/bin/clang
CXX=/opt/rocm/llvm/bin/clang++
export CC CXX
PYTHONPATH="$runtime_dir/plugins${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
"#,
        );
    }
    environment
}

fn write_rocm_platform_plugin(wsl: &WslContext, runtime: &str) -> Result<()> {
    let plugin_root = format!("{runtime}/plugins");
    let metadata_root =
        format!("{plugin_root}/omniinfer_vllm_wsl2_rocm-{ROCM_PLATFORM_PLUGIN_VERSION}.dist-info");
    let metadata = format!(
        "Metadata-Version: 2.1\n\
         Name: omniinfer-vllm-wsl2-rocm\n\
         Version: {ROCM_PLATFORM_PLUGIN_VERSION}\n\
         Summary: OmniInfer vLLM ROCm platform detection for supported WSL2 GPUs\n"
    );
    write_wsl_file(
        wsl,
        &format!("{plugin_root}/omniinfer_vllm_wsl2_rocm.py"),
        ROCM_PLATFORM_PLUGIN.as_bytes(),
        false,
    )?;
    write_wsl_file(
        wsl,
        &format!("{metadata_root}/METADATA"),
        metadata.as_bytes(),
        false,
    )?;
    write_wsl_file(
        wsl,
        &format!("{metadata_root}/entry_points.txt"),
        ROCM_PLATFORM_PLUGIN_ENTRY_POINTS.as_bytes(),
        false,
    )?;
    Ok(())
}

fn query_nvidia_driver() -> Result<String> {
    let executable = std::env::var_os("OMNIINFER_VLLM_NVIDIA_SMI")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("nvidia-smi"));
    let output = run_command(
        &executable,
        ["--query-gpu=driver_version", "--format=csv,noheader"],
        None,
    )
    .with_context(|| format!("run {}", executable.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "nvidia-smi failed while detecting the NVIDIA driver: {}",
            decode_output(&output.stderr).trim()
        );
    }
    decode_output(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("nvidia-smi returned no NVIDIA driver version"))
}

fn require_minimum_driver(actual: &str, minimum: &str) -> Result<()> {
    let actual_version = parse_version(actual)
        .ok_or_else(|| anyhow::anyhow!("invalid NVIDIA driver version: {actual}"))?;
    let minimum_version = parse_version(minimum)
        .ok_or_else(|| anyhow::anyhow!("invalid catalog minimum driver version: {minimum}"))?;
    if actual_version < minimum_version {
        anyhow::bail!(
            "NVIDIA driver {actual} is too old for the pinned vLLM CUDA runtime; minimum required driver is {minimum}"
        );
    }
    Ok(())
}

fn parse_version(value: &str) -> Option<(u32, u32, u32)> {
    let mut parts = value.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

fn validate_installed_runtime(
    wsl: &WslContext,
    linux_current: &str,
    expected_version: &str,
    accelerator: &str,
    expected_runtime: &str,
    reporter: &mut InstallReporter,
) -> Result<Value> {
    ensure_runtime_not_active(wsl, linux_current)?;
    validate_runtime_path(
        wsl,
        linux_current,
        expected_version,
        accelerator,
        expected_runtime,
        reporter,
    )
}

fn validate_runtime_path(
    wsl: &WslContext,
    runtime: &str,
    expected_version: &str,
    accelerator: &str,
    expected_runtime: &str,
    reporter: &mut InstallReporter,
) -> Result<Value> {
    let python = format!("{runtime}/venv/bin/python");
    validate_native_dependencies(wsl, runtime, reporter)?;
    let script = r#"set -eu
runtime=$1
runtime_dir=$runtime
python=$2
probe=$3
accelerator=$4
set -a
. "$runtime/runtime.env"
set +a
export OMNIINFER_EXPECTED_ACCELERATOR=$accelerator
exec "$python" -c "$probe"
"#;
    let output = run_wsl(
        wsl,
        [
            "sh",
            "-c",
            script,
            "sh",
            runtime,
            &python,
            GPU_PROBE,
            accelerator,
        ],
        None,
    )
    .with_context(|| format!("validate managed vLLM runtime {runtime}"))?;
    append_install_log(&wsl.install_log, "gpu-probe", &output);
    if !output.status.success() {
        anyhow::bail!(
            "managed vLLM GPU validation failed: {}",
            decode_output(&output.stderr).trim()
        );
    }
    let stdout = decode_output(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow::anyhow!("managed vLLM GPU probe returned no JSON result"))?;
    let probe: Value = serde_json::from_str(line).context("parse managed vLLM GPU probe")?;
    if probe["vllm_version"].as_str() != Some(expected_version) {
        anyhow::bail!(
            "managed vLLM version mismatch: expected {expected_version}, got {}",
            probe["vllm_version"]
        );
    }
    let (field, label) = if accelerator == "rocm" {
        ("torch_hip", "ROCm")
    } else {
        ("torch_cuda", "CUDA")
    };
    if probe[field].as_str() != Some(expected_runtime) {
        anyhow::bail!(
            "managed vLLM {label} ABI mismatch: expected {expected_runtime}, got {}",
            probe[field]
        );
    }
    reporter.event(
        "validation_passed",
        json!({
            "distribution": wsl.distribution,
            "runtime": runtime,
            "probe": probe,
        }),
    );
    Ok(probe)
}

fn validate_native_dependencies(
    wsl: &WslContext,
    runtime: &str,
    reporter: &mut InstallReporter,
) -> Result<()> {
    let output = run_wsl(
        wsl,
        ["sh", "-c", NATIVE_DEPENDENCY_PROBE, "sh", runtime],
        None,
    )
    .with_context(|| format!("validate managed native dependencies {runtime}"))?;
    append_install_log(&wsl.install_log, "native-dependencies", &output);
    if !output.status.success() {
        anyhow::bail!(
            "managed vLLM native dependency validation failed: {}",
            decode_output(&output.stderr).trim()
        );
    }
    let checked = decode_output(&output.stdout)
        .lines()
        .rev()
        .find_map(|line| line.trim().parse::<u64>().ok())
        .ok_or_else(|| anyhow::anyhow!("managed native dependency probe returned no count"))?;
    reporter.event(
        "native_dependencies_verified",
        json!({
            "distribution": wsl.distribution,
            "runtime": runtime,
            "extensions_checked": checked,
        }),
    );
    Ok(())
}

fn ensure_runtime_not_active(wsl: &WslContext, linux_current: &str) -> Result<()> {
    let script = r#"set -eu
runtime=$1
found=0
for pid_file in "$runtime"/run/*.pid; do
    [ -e "$pid_file" ] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    case "$pid" in ''|*[!0-9]*) rm -f "$pid_file"; continue;; esac
    if kill -0 "$pid" 2>/dev/null; then
        echo "$pid_file:$pid"
        found=1
    else
        rm -f "$pid_file"
    fi
done
exit "$found"
"#;
    let output = run_wsl(wsl, ["sh", "-c", script, "sh", linux_current], None)?;
    match output.status.code() {
        Some(0) => Ok(()),
        Some(1) => anyhow::bail!(
            "vLLM WSL2 runtime is active ({}); unload the model or stop OmniInfer before reinstalling",
            decode_output(&output.stdout).trim()
        ),
        _ => anyhow::bail!(
            "failed to inspect active WSL2 runtime: {}",
            decode_output(&output.stderr).trim()
        ),
    }
}

fn activate_runtime(
    wsl: &WslContext,
    base: &str,
    staging: &str,
    current: &str,
    backup: &str,
    reporter: &mut InstallReporter,
) -> Result<()> {
    let script = r#"set -eu
base=$1
staging=$2
current=$3
backup=$4
test -d "$staging"
rm -rf "$backup"
if [ -e "$current" ]; then
    mv "$current" "$backup"
fi
if ! mv "$staging" "$current"; then
    if [ -e "$backup" ] && [ ! -e "$current" ]; then
        mv "$backup" "$current"
    fi
    exit 1
fi
sync
"#;
    run_wsl_checked(
        wsl,
        ["sh", "-c", script, "sh", base, staging, current, backup],
        reporter,
        "activate managed WSL runtime",
    )
}

fn rollback_runtime(
    wsl: &WslContext,
    current: &str,
    backup: &str,
    reporter: &mut InstallReporter,
) -> Result<()> {
    let script = r#"set -eu
current=$1
backup=$2
rm -rf "$current"
if [ -e "$backup" ]; then
    mv "$backup" "$current"
fi
"#;
    run_wsl_checked(
        wsl,
        ["sh", "-c", script, "sh", current, backup],
        reporter,
        "roll back managed WSL runtime",
    )
}

fn write_wsl_file(wsl: &WslContext, path: &str, bytes: &[u8], executable: bool) -> Result<()> {
    let parent = path
        .rsplit_once('/')
        .map(|(parent, _)| parent)
        .ok_or_else(|| anyhow::anyhow!("invalid Linux runtime path: {path}"))?;
    let mkdir = run_wsl(wsl, ["mkdir", "-p", parent], None)?;
    require_success(&mkdir, "create WSL file parent")?;
    let output = run_wsl(wsl, ["tee", path], Some(bytes))?;
    require_success(&output, "write WSL runtime file")?;
    if executable {
        let chmod = run_wsl(wsl, ["chmod", "0755", path], None)?;
        require_success(&chmod, "mark WSL runtime file executable")?;
    }
    Ok(())
}

fn write_launcher_manifest(runtime_dir: &Path, manifest: &LauncherManifest) -> Result<()> {
    let bin_dir = runtime_dir.join("bin");
    fs::create_dir_all(&bin_dir)
        .with_context(|| format!("create launcher directory {}", bin_dir.display()))?;
    let target = bin_dir.join(LAUNCHER_MANIFEST);
    let temporary = bin_dir.join(format!("{LAUNCHER_MANIFEST}.tmp-{}", unique_suffix()));
    let bytes = serde_json::to_vec_pretty(manifest)?;
    {
        let mut file = File::create(&temporary)
            .with_context(|| format!("create launcher manifest {}", temporary.display()))?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&temporary, &target)
        .with_context(|| format!("activate launcher manifest {}", target.display()))?;
    Ok(())
}

fn launcher_manifest_matches(runtime_dir: &Path, expected: &LauncherManifest) -> bool {
    let path = runtime_dir.join("bin").join(LAUNCHER_MANIFEST);
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<LauncherManifest>(&raw).ok())
        .is_some_and(|actual| {
            actual.schema_version == expected.schema_version
                && actual.backend == expected.backend
                && actual.distribution == expected.distribution
                && actual.source == expected.source
                && actual.tag == expected.tag
                && actual.python == expected.python
                && actual.uv_version == expected.uv_version
                && actual.uv_sha256 == expected.uv_sha256
                && actual.package_version == expected.package_version
                && actual.wheel_sha256 == expected.wheel_sha256
                && actual.accelerator == expected.accelerator
                && actual.runtime_version == expected.runtime_version
                && actual.runtime_environment_version == expected.runtime_environment_version
                && actual.linux_launcher == expected.linux_launcher
                && actual.linux_runner == expected.linux_runner
                && actual.linux_stopper == expected.linux_stopper
                && actual.linux_pid_dir == expected.linux_pid_dir
                && actual.automount_root == expected.automount_root
        })
}

fn extract_uv(runtime_dir: &Path, archive: &[u8]) -> Result<PathBuf> {
    let tools = runtime_dir.join("tools");
    fs::create_dir_all(&tools)?;
    let target = tools.join("uv-linux-x86_64");
    let decoder = GzDecoder::new(Cursor::new(archive));
    let mut tar = tar::Archive::new(decoder);
    let mut found = false;
    for entry in tar.entries().context("read managed uv archive")? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if entry.header().entry_type().is_file()
            && path.file_name().and_then(|name| name.to_str()) == Some("uv")
        {
            let mut file = File::create(&target)?;
            std::io::copy(&mut entry, &mut file)?;
            file.sync_all()?;
            found = true;
            break;
        }
    }
    if !found {
        anyhow::bail!("managed uv archive does not contain the uv executable");
    }
    Ok(target)
}

fn acquire_install_lock(runtime_dir: &Path) -> Result<InstallLock> {
    let path = runtime_dir.join(".install.lock");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| {
            format!(
                "acquire WSL runtime install lock {}; another install may be active",
                path.display()
            )
        })?;
    Ok(InstallLock { path })
}

fn wsl_path(wsl: &WslContext, windows_path: &Path) -> Result<String> {
    let text = windows_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Windows runtime path is not valid UTF-8"))?;
    run_wsl_text(
        &wsl.executable,
        &wsl.distribution,
        ["wslpath", "-a", "-u", text],
    )
}

fn run_wsl_checked<I, S>(
    wsl: &WslContext,
    args: I,
    reporter: &mut InstallReporter,
    action: &str,
) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    run_wsl_as_checked(wsl, None, args, None, reporter, action)
}

fn run_wsl_as_checked<I, S>(
    wsl: &WslContext,
    user: Option<&str>,
    args: I,
    stdin: Option<&[u8]>,
    reporter: &mut InstallReporter,
    action: &str,
) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args = args
        .into_iter()
        .map(|value| value.as_ref().to_string())
        .collect::<Vec<_>>();
    reporter.event(
        "command_started",
        json!({
            "action": action,
            "distribution": wsl.distribution,
            "command": args.first(),
            "user": user,
        }),
    );
    let output = run_wsl_as(wsl, user, args.iter().map(String::as_str), stdin)
        .with_context(|| action.to_string())?;
    append_install_log(&wsl.install_log, action, &output);
    require_success(&output, action)?;
    reporter.event("command_completed", json!({ "action": action }));
    Ok(())
}

fn run_wsl_as_file_checked<I, S>(
    wsl: &WslContext,
    user: Option<&str>,
    args: I,
    stdin_path: &Path,
    reporter: &mut InstallReporter,
    action: &str,
) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args = args
        .into_iter()
        .map(|value| value.as_ref().to_string())
        .collect::<Vec<_>>();
    reporter.event(
        "command_started",
        json!({
            "action": action,
            "distribution": wsl.distribution,
            "command": args.first(),
            "user": user,
        }),
    );
    let output = run_wsl_as_file(wsl, user, args.iter().map(String::as_str), stdin_path)
        .with_context(|| action.to_string())?;
    append_install_log(&wsl.install_log, action, &output);
    require_success(&output, action)?;
    reporter.event("command_completed", json!({ "action": action }));
    Ok(())
}

fn run_wsl<I, S>(wsl: &WslContext, args: I, stdin: Option<&[u8]>) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    run_wsl_as(wsl, None, args, stdin)
}

fn run_wsl_as<I, S>(
    wsl: &WslContext,
    user: Option<&str>,
    args: I,
    stdin: Option<&[u8]>,
) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut full_args = vec!["--distribution".to_string(), wsl.distribution.clone()];
    if let Some(user) = user {
        full_args.extend(["--user".to_string(), user.to_string()]);
    }
    full_args.push("--exec".to_string());
    full_args.extend(args.into_iter().map(|value| value.as_ref().to_string()));
    run_command(&wsl.executable, full_args.iter().map(String::as_str), stdin)
}

fn run_wsl_as_file<I, S>(
    wsl: &WslContext,
    user: Option<&str>,
    args: I,
    stdin_path: &Path,
) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut full_args = vec!["--distribution".to_string(), wsl.distribution.clone()];
    if let Some(user) = user {
        full_args.extend(["--user".to_string(), user.to_string()]);
    }
    full_args.push("--exec".to_string());
    full_args.extend(args.into_iter().map(|value| value.as_ref().to_string()));
    let stdin = File::open(stdin_path)
        .with_context(|| format!("open staged input {}", stdin_path.display()))?;
    let mut command = Command::new(&wsl.executable);
    command
        .args(full_args)
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_child_window(&mut command);
    command
        .output()
        .with_context(|| format!("run {}", wsl.executable.display()))
}

fn run_wsl_text<I, S>(executable: &Path, distro: &str, args: I) -> Result<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut full_args = vec![
        "--distribution".to_string(),
        distro.to_string(),
        "--exec".to_string(),
    ];
    full_args.extend(args.into_iter().map(|value| value.as_ref().to_string()));
    let output = run_command(executable, full_args.iter().map(String::as_str), None)?;
    require_success(&output, "query WSL2 distribution")?;
    Ok(decode_output(&output.stdout).trim().to_string())
}

fn run_command<I, S>(executable: &Path, args: I, stdin: Option<&[u8]>) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut command = Command::new(executable);
    command
        .args(args.into_iter().map(|value| value.as_ref().to_string()))
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_child_window(&mut command);
    let mut child = command
        .spawn()
        .with_context(|| format!("start {}", executable.display()))?;
    if let Some(bytes) = stdin
        && let Some(mut stream) = child.stdin.take()
    {
        stream.write_all(bytes)?;
    }
    child
        .wait_with_output()
        .with_context(|| format!("wait for {}", executable.display()))
}

fn require_success(output: &Output, action: &str) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }
    anyhow::bail!(
        "{action} failed with {}: {}",
        output.status,
        decode_output(&output.stderr).trim()
    )
}

fn decode_output(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes.iter().skip(1).step_by(2).any(|byte| *byte == 0) {
        let words = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        String::from_utf16_lossy(&words)
            .trim_start_matches('\u{feff}')
            .to_string()
    } else {
        String::from_utf8_lossy(bytes).to_string()
    }
}

fn distro_is_wsl2(name: &str, verbose: &str) -> bool {
    verbose.lines().any(|line| {
        let normalized = line.trim().trim_start_matches('*').trim();
        let distro = normalized.split_whitespace().next();
        let Some(version) = normalized.split_whitespace().last() else {
            return false;
        };
        version == "2" && distro.is_some_and(|distro| distro.eq_ignore_ascii_case(name))
    })
}

fn default_distro(verbose: &str) -> Option<String> {
    verbose.lines().find_map(|line| {
        let trimmed = line.trim_start();
        let rest = trimmed.strip_prefix('*')?.trim_start();
        let version = rest.split_whitespace().last()?;
        (version == "2")
            .then(|| rest.split_whitespace().next().map(str::to_string))
            .flatten()
    })
}

fn is_internal_distro(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower == "docker-desktop" || lower == "docker-desktop-data" || lower.ends_with("-data")
}

fn automount_root_from_c_path(c_root: &str) -> Option<String> {
    let (root, drive) = c_root.trim().trim_end_matches('/').rsplit_once('/')?;
    if !drive.eq_ignore_ascii_case("c") {
        return None;
    }
    if root.is_empty() {
        Some("/".to_string())
    } else if root.starts_with('/') {
        Some(root.to_string())
    } else {
        None
    }
}

fn eligible_distro_names(names: &[String], verbose: &str) -> Vec<String> {
    names
        .iter()
        .filter(|name| !is_internal_distro(name) && distro_is_wsl2(name, verbose))
        .cloned()
        .collect()
}

fn runtime_key(runtime_dir: &Path) -> String {
    let normalized = runtime_dir
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    let digest = Sha256::digest(normalized.as_bytes());
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn unique_suffix() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .to_string()
}

fn append_install_log(path: &Path, action: &str, output: &Output) {
    if path.as_os_str().is_empty() {
        return;
    }
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let _ = writeln!(file, "\n== {action} ==");
    let stdout = decode_output(&output.stdout);
    if !stdout.trim().is_empty() {
        let _ = writeln!(file, "stdout:\n{}", stdout.trim_end());
    }
    let stderr = decode_output(&output.stderr);
    if !stderr.trim().is_empty() {
        let _ = writeln!(file, "stderr:\n{}", stderr.trim_end());
    }
    let _ = writeln!(file, "status: {}", output.status);
}

fn hide_child_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_utf16_wsl_output() {
        let text = "Ubuntu\r\n";
        let bytes = text
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(decode_output(&bytes), text);
    }

    #[test]
    fn rejects_internal_distributions() {
        assert!(is_internal_distro("docker-desktop"));
        assert!(is_internal_distro("example-data"));
        assert!(!is_internal_distro("Ubuntu-24.04"));
    }

    #[test]
    fn selects_wsl2_rows_without_localized_headers() {
        let rows = "  NAME STATE VERSION\r\n* Ubuntu Running 2\r\n  Legacy Stopped 1\r\n";
        assert!(distro_is_wsl2("Ubuntu", rows));
        assert!(!distro_is_wsl2("Legacy", rows));
        assert_eq!(default_distro(rows).as_deref(), Some("Ubuntu"));
    }

    #[test]
    fn eligible_distributions_exclude_wsl1_and_internal_distros() {
        let names = vec![
            "Ubuntu".to_string(),
            "Legacy".to_string(),
            "docker-desktop".to_string(),
            "docker-desktop-data".to_string(),
        ];
        let rows = "  NAME STATE VERSION\r\n* Ubuntu Running 2\r\n  Legacy Stopped 1\r\n  docker-desktop Running 2\r\n  docker-desktop-data Stopped 2\r\n";
        assert_eq!(eligible_distro_names(&names, rows), vec!["Ubuntu"]);
    }

    #[test]
    fn minimum_driver_check_uses_windows_cuda_requirement() {
        assert!(require_minimum_driver("581.57", "576.02").is_ok());
        let error = require_minimum_driver("575.99", "576.02").unwrap_err();
        assert!(error.to_string().contains("too old"));
    }

    #[test]
    fn runtime_environment_exposes_managed_native_libraries() {
        let cuda = runtime_environment("cuda");
        assert!(cuda.contains("site-packages/torch/lib"));
        assert!(cuda.contains("site-packages/nvidia/*/lib"));
        assert!(cuda.contains("export LD_LIBRARY_PATH"));
        assert!(!cuda.contains("HSA_ENABLE_DXG_DETECTION"));

        let rocm = runtime_environment("rocm");
        assert!(rocm.contains("site-packages/torch/lib"));
        assert!(rocm.contains("HSA_ENABLE_DXG_DETECTION=1"));
        assert!(rocm.contains("CC=/opt/rocm/llvm/bin/clang"));
        assert!(rocm.contains("export CC CXX"));
        assert!(rocm.contains(r#"PYTHONPATH="$runtime_dir/plugins"#));
    }

    #[test]
    fn runner_limits_default_rocm_kv_cache_without_overriding_explicit_policy() {
        assert!(RUNNER_SCRIPT.contains(r#"HSA_ENABLE_DXG_DETECTION:-}" = "1"#));
        assert!(RUNNER_SCRIPT.contains("--kv-cache-memory-bytes"));
        assert!(RUNNER_SCRIPT.contains("--gpu-memory-utilization"));
        assert!(RUNNER_SCRIPT.contains("memory_kib * 1024 / 5"));
        assert!(RUNNER_SCRIPT.contains("4294967296"));
    }

    #[test]
    fn runner_applies_overridable_wsl2_rocm_execution_defaults() {
        assert!(RUNNER_SCRIPT.contains("--enforce-eager|--no-enforce-eager"));
        assert!(RUNNER_SCRIPT.contains(
            "--enable-chunked-prefill|--enable-chunked-prefill=*|--no-enable-chunked-prefill"
        ));
        assert!(RUNNER_SCRIPT.contains(r#"set -- "$@" --enforce-eager"#));
        assert!(RUNNER_SCRIPT.contains(r#"set -- "$@" --no-enable-chunked-prefill"#));
    }

    #[test]
    fn rocm_platform_plugin_uses_the_upstream_extension_point() {
        assert!(ROCM_PLATFORM_PLUGIN.contains("torch.cuda.is_available()"));
        assert!(ROCM_PLATFORM_PLUGIN.contains("_amdsmi_has_gpu()"));
        assert!(ROCM_PLATFORM_PLUGIN.contains("_install_amdsmi_shim(devices)"));
        assert!(ROCM_PLATFORM_PLUGIN.contains("class WslRocmBuffer"));
        assert!(ROCM_PLATFORM_PLUGIN.contains("copy_to_accelerator"));
        assert!(ROCM_PLATFORM_PLUGIN_ENTRY_POINTS.contains("[vllm.platform_plugins]"));
        assert!(ROCM_PLATFORM_PLUGIN_ENTRY_POINTS.contains("[vllm.general_plugins]"));
        assert!(
            ROCM_PLATFORM_PLUGIN_ENTRY_POINTS.contains("omniinfer_vllm_wsl2_rocm:platform_plugin")
        );
    }

    #[test]
    fn derives_default_and_root_level_wsl_automounts() {
        assert_eq!(
            automount_root_from_c_path("/mnt/c/").as_deref(),
            Some("/mnt")
        );
        assert_eq!(automount_root_from_c_path("/c").as_deref(), Some("/"));
        assert_eq!(automount_root_from_c_path("/mnt/d"), None);
    }
}
