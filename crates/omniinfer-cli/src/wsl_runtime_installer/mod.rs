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

mod assets;
mod rocm;
mod wsl;

use assets::{
    GPU_PROBE, NATIVE_DEPENDENCY_PROBE, ROCM_PLATFORM_PLUGIN, ROCM_PLATFORM_PLUGIN_ENTRY_POINTS,
    RUNNER_SCRIPT, STOPPER_SCRIPT,
};
use rocm::{
    ensure_rocm_system_runtime, runtime_environment, validate_existing_system_runtime,
    validate_rocm_distro, validate_wsl_rocm_gpu, write_rocm_platform_plugin,
};
use wsl::*;
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
