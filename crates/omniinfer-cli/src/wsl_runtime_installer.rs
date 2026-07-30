use std::fs::{self, File, OpenOptions};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::backend_installer::{InstallReporter, download_verified_asset};
use crate::prebuilt_catalog::{PrebuiltCatalog, PythonRuntimeEntry};

const BACKEND_ID: &str = "vllm-wsl2-cuda";
const LAUNCHER_MANIFEST: &str = "vllm-wsl2.json";
const MANAGED_MANIFEST: &str = "managed-runtime.json";

const RUNNER_SCRIPT: &str = r#"#!/bin/sh
set -eu
pid_file=$1
shift
mkdir -p "$(dirname "$pid_file")"
if [ -s "$pid_file" ]; then
    old_pid=$(cat "$pid_file")
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "vLLM runtime is already active with pid $old_pid" >&2
        exit 73
    fi
    rm -f "$pid_file"
fi
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
import torch
import vllm
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print(json.dumps({
    "vllm_version": vllm.__version__,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "value": float(x.item()),
}))
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
    if backend != BACKEND_ID {
        anyhow::bail!("unsupported managed WSL2 backend: {backend}");
    }
    if std::env::consts::OS != "windows"
        && std::env::var_os("OMNIINFER_TEST_WSL_PLATFORM").is_none()
    {
        anyhow::bail!("vllm-wsl2-cuda is supported only by the Windows OmniInfer CLI");
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
    let driver = query_nvidia_driver()?;
    require_minimum_driver(&driver, &variant.minimum_driver)?;
    let mut wsl = detect_wsl_context(requested_distro)?;
    wsl.install_log = runtime_dir.join("logs").join("install.log");
    validate_wsl_gpu(&wsl)?;
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
    };

    reporter.human(format!("Managed WSL2 Python runtime: windows/{backend}"));
    reporter.human(format!("  distribution: {}", wsl.distribution));
    reporter.human(format!("  Windows runtime: {}", runtime_dir.display()));
    reporter.human(format!("  Linux runtime: {linux_current}"));
    reporter.human(format!("  NVIDIA driver: {driver}"));
    reporter.human(format!(
        "  selected CUDA ABI: {} ({})",
        variant.cuda, variant.torch_backend
    ));
    reporter.human(format!("  wheel sha256: {}", variant.sha256));
    reporter.event(
        "compatibility_selected",
        json!({
            "architecture": architecture,
            "distribution": wsl.distribution,
            "driver": driver,
            "minimum_driver": variant.minimum_driver,
            "cuda": variant.cuda,
            "torch_backend": variant.torch_backend,
            "wheel_url": variant.url,
            "package_version": variant.version,
            "wheel_sha256": variant.sha256,
            "linux_runtime": linux_current,
        }),
    );

    if launcher_manifest_matches(runtime_dir, &expected)
        && validate_installed_runtime(
            &wsl,
            &linux_current,
            &variant.version,
            &variant.cuda,
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
        reporter.event(
            "dry_run_completed",
            json!({
                "runtime_dir": runtime_dir,
                "distribution": wsl.distribution,
                "linux_runtime": linux_current,
                "wheel_url": variant.url,
                "package_version": variant.version,
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
        run_wsl_checked(
            &wsl,
            [
                linux_uv.as_str(),
                "pip",
                "install",
                "--python",
                format!("{linux_staging}/venv/bin/python").as_str(),
                "--torch-backend",
                variant.torch_backend.as_str(),
                "--index-strategy",
                "first-index",
                requirement.as_str(),
            ],
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
            "wheel_sha256": variant.sha256,
            "cuda": variant.cuda,
            "torch_backend": variant.torch_backend,
            "minimum_driver": variant.minimum_driver,
            "driver": driver,
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
            &variant.version,
            &variant.cuda,
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
            &variant.version,
            &variant.cuda,
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
            "WSL is unavailable. Enable WSL2 and install Ubuntu before installing {BACKEND_ID}: {}",
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
            "{BACKEND_ID} requires an x86_64 WSL2 distribution, found {}",
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

fn validate_wsl_gpu(wsl: &WslContext) -> Result<()> {
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
    expected_cuda: &str,
    reporter: &mut InstallReporter,
) -> Result<Value> {
    ensure_runtime_not_active(wsl, linux_current)?;
    validate_runtime_path(
        wsl,
        linux_current,
        expected_version,
        expected_cuda,
        reporter,
    )
}

fn validate_runtime_path(
    wsl: &WslContext,
    runtime: &str,
    expected_version: &str,
    expected_cuda: &str,
    reporter: &mut InstallReporter,
) -> Result<Value> {
    let python = format!("{runtime}/venv/bin/python");
    let output = run_wsl(wsl, [python.as_str(), "-c", GPU_PROBE], None)
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
    if probe["torch_cuda"].as_str() != Some(expected_cuda) {
        anyhow::bail!(
            "managed vLLM CUDA ABI mismatch: expected {expected_cuda}, got {}",
            probe["torch_cuda"]
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
        }),
    );
    let output =
        run_wsl(wsl, args.iter().map(String::as_str), None).with_context(|| action.to_string())?;
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
    let mut full_args = vec![
        "--distribution".to_string(),
        wsl.distribution.clone(),
        "--exec".to_string(),
    ];
    full_args.extend(args.into_iter().map(|value| value.as_ref().to_string()));
    run_command(&wsl.executable, full_args.iter().map(String::as_str), stdin)
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
    fn derives_default_and_root_level_wsl_automounts() {
        assert_eq!(
            automount_root_from_c_path("/mnt/c/").as_deref(),
            Some("/mnt")
        );
        assert_eq!(automount_root_from_c_path("/c").as_deref(), Some("/"));
        assert_eq!(automount_root_from_c_path("/mnt/d"), None);
    }
}
