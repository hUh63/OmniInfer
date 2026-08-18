use super::*;

pub(super) fn validate_rocm_distro(wsl: &WslContext) -> Result<()> {
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

pub(super) fn ensure_rocm_system_runtime(
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

pub(super) fn validate_existing_system_runtime(
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

pub(super) fn validate_wsl_rocm_gpu(wsl: &WslContext, system: &RocmSystemRuntime) -> Result<()> {
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

pub(super) fn runtime_environment(accelerator: &str) -> String {
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

pub(super) fn write_rocm_platform_plugin(wsl: &WslContext, runtime: &str) -> Result<()> {
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
