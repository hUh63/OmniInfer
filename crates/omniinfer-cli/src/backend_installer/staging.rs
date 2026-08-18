use super::*;

pub(super) fn prepare_and_install_runtime(
    work_dir: &Path,
    runtime_dir: &Path,
    entry: &PrebuiltEntry,
    archives: &[DownloadedArchive],
) -> Result<PathBuf> {
    let primary = archives
        .first()
        .ok_or_else(|| anyhow::anyhow!("prebuilt catalog produced no runtime archive"))?;
    let primary_dir = work_dir.join("asset-0");
    fs::create_dir_all(&primary_dir)?;
    extract_archive(&primary.bytes, &primary.archive, &primary_dir)?;
    let launcher = find_launcher(&primary_dir, &entry.launcher)?;
    let source_dir = launcher
        .parent()
        .ok_or_else(|| anyhow::anyhow!("launcher has no parent directory"))?;
    let staged_bin = work_dir.join("staged-bin");
    fs::create_dir_all(&staged_bin)?;
    copy_directory_contents(source_dir, &staged_bin)?;

    if archives.len() != 1 + entry.companion_assets.len() {
        anyhow::bail!("downloaded prebuilt asset count does not match the catalog");
    }
    for (index, asset) in entry.companion_assets.iter().enumerate() {
        let archive = &archives[index + 1];
        let asset_dir = work_dir.join(format!("asset-{}", index + 1));
        fs::create_dir_all(&asset_dir)?;
        extract_archive(&archive.bytes, &archive.archive, &asset_dir)?;
        if asset.files.is_empty() {
            anyhow::bail!("companion asset {} does not declare any files", index + 1);
        }
        for file in &asset.files {
            copy_named_asset_file(&asset_dir, &staged_bin, file)?;
        }
    }
    validate_required_runtime_files(&staged_bin, entry)?;

    install_staged_runtime(&staged_bin, runtime_dir, &entry.launcher)
}

pub(super) fn install_staged_runtime(
    staged_bin: &Path,
    runtime_dir: &Path,
    launcher_name: &str,
) -> Result<PathBuf> {
    let bin_dir = runtime_dir.join("bin");
    let logs_dir = runtime_dir.join("logs");
    fs::create_dir_all(&logs_dir)?;
    let suffix = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let next_dir = runtime_dir.join(format!("bin.installing-{suffix}"));
    let backup_dir = runtime_dir.join(format!("bin.backup-{suffix}"));
    copy_dir_recursive(staged_bin, &next_dir)?;
    let next_launcher = next_dir.join(launcher_name);
    if !next_launcher.is_file() {
        let _ = fs::remove_dir_all(&next_dir);
        anyhow::bail!(
            "prebuilt install failed: {} was not staged",
            next_launcher.display()
        );
    }
    make_executable(&next_launcher)?;

    if bin_dir.exists() {
        fs::rename(&bin_dir, &backup_dir).with_context(|| {
            format!(
                "move existing runtime {} to {}",
                bin_dir.display(),
                backup_dir.display()
            )
        })?;
    }
    if let Err(error) = fs::rename(&next_dir, &bin_dir) {
        if backup_dir.exists() {
            let _ = fs::rename(&backup_dir, &bin_dir);
        }
        let _ = fs::remove_dir_all(&next_dir);
        return Err(error).with_context(|| {
            format!(
                "activate staged runtime {} as {}",
                next_dir.display(),
                bin_dir.display()
            )
        });
    }
    if backup_dir.exists()
        && let Err(error) = fs::remove_dir_all(&backup_dir)
    {
        eprintln!(
            "warning: failed to remove old runtime backup {}: {error}",
            backup_dir.display()
        );
    }

    let installed_launcher = bin_dir.join(launcher_name);
    if !installed_launcher.is_file() {
        anyhow::bail!(
            "prebuilt install failed: {} was not created",
            installed_launcher.display()
        );
    }
    make_executable(&installed_launcher)?;
    Ok(installed_launcher)
}

pub(super) fn copy_named_asset_file(
    source_root: &Path,
    target_root: &Path,
    file: &str,
) -> Result<()> {
    let relative = Path::new(file);
    validate_archive_path(relative)?;
    if relative.components().count() != 1 {
        anyhow::bail!("companion asset file must be a plain file name: {file}");
    }
    let source = find_launcher(source_root, file)
        .with_context(|| format!("required companion file {file} was not found"))?;
    let target = target_root.join(file);
    fs::copy(&source, &target).with_context(|| {
        format!(
            "copy companion file {} to {}",
            source.display(),
            target.display()
        )
    })?;
    Ok(())
}

pub(super) fn missing_required_runtime_files(
    runtime_dir: &Path,
    entry: &PrebuiltEntry,
) -> Result<Vec<String>> {
    let bin_dir = runtime_dir.join("bin");
    let mut required = Vec::with_capacity(1 + entry.required_files.len());
    required.push(entry.launcher.as_str());
    required.extend(entry.required_files.iter().map(String::as_str));
    let mut missing = Vec::new();
    for file in required {
        let relative = Path::new(file);
        validate_archive_path(relative)?;
        if !bin_dir.join(relative).is_file() {
            missing.push(file.to_string());
        }
    }
    Ok(missing)
}

pub(super) fn existing_prebuilt_verification_failure(
    runtime_dir: &Path,
    entry: &PrebuiltEntry,
) -> Option<String> {
    let manifest_path = runtime_dir.join("prebuilt.json");
    if !manifest_path.is_file() {
        return None;
    }
    let raw = match fs::read_to_string(&manifest_path) {
        Ok(raw) => raw,
        Err(error) => return Some(format!("cannot read prebuilt manifest: {error}")),
    };
    let manifest: Value = match serde_json::from_str(&raw) {
        Ok(manifest) => manifest,
        Err(error) => return Some(format!("cannot parse prebuilt manifest: {error}")),
    };
    if let Some(expected) = entry.sha256.as_deref() {
        let actual = manifest
            .get("archive_sha256")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !expected.eq_ignore_ascii_case(actual) {
            return Some(format!(
                "runtime archive digest is {actual:?}, expected {expected}"
            ));
        }
    }
    let assets = manifest.get("assets").and_then(Value::as_array);
    for (index, companion) in entry.companion_assets.iter().enumerate() {
        let Some(expected) = companion.sha256.as_deref() else {
            continue;
        };
        let actual = assets
            .and_then(|items| items.get(index + 1))
            .and_then(|asset| asset.get("archive_sha256"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if !expected.eq_ignore_ascii_case(actual) {
            return Some(format!(
                "companion {} archive digest is {actual:?}, expected {expected}",
                index + 1
            ));
        }
    }
    None
}

pub(super) fn validate_required_runtime_files(bin_dir: &Path, entry: &PrebuiltEntry) -> Result<()> {
    let mut required = Vec::with_capacity(1 + entry.required_files.len());
    required.push(entry.launcher.as_str());
    required.extend(entry.required_files.iter().map(String::as_str));
    let mut missing = Vec::new();
    for file in required {
        let relative = Path::new(file);
        validate_archive_path(relative)?;
        if !bin_dir.join(relative).is_file() {
            missing.push(file);
        }
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "prebuilt install is incomplete; missing required files: {}",
            missing.join(", ")
        );
    }
    Ok(())
}

pub(super) fn find_launcher(root: &Path, launcher: &str) -> Result<PathBuf> {
    let mut matches = Vec::new();
    collect_launcher_matches(root, launcher, &mut matches)?;
    matches.sort_by(|left, right| {
        left.components()
            .count()
            .cmp(&right.components().count())
            .then_with(|| left.cmp(right))
    });
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("launcher {launcher:?} was not found in extracted archive"))
}

pub(super) fn collect_launcher_matches(
    root: &Path,
    launcher: &str,
    matches: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_launcher_matches(&path, launcher, matches)?;
        } else if file_type.is_file()
            && path.file_name().and_then(|value| value.to_str()) == Some(launcher)
        {
            matches.push(path);
        }
    }
    Ok(())
}

pub(super) fn copy_directory_contents(source_dir: &Path, target_dir: &Path) -> Result<()> {
    let source_root = fs::canonicalize(source_dir)
        .with_context(|| format!("resolve copy source {}", source_dir.display()))?;
    copy_directory_contents_in_root(source_dir, target_dir, &source_root)
}

pub(super) fn copy_directory_contents_in_root(
    source_dir: &Path,
    target_dir: &Path,
    source_root: &Path,
) -> Result<()> {
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let source = entry.path();
        let target = target_dir.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            copy_safe_symbolic_link(&source, &target, source_root)?;
        } else if file_type.is_dir() {
            fs::create_dir_all(&target)?;
            copy_directory_contents_in_root(&source, &target, source_root)?;
        } else if file_type.is_file() {
            fs::copy(&source, &target)?;
            make_executable_if_source_is_executable(&source, &target)?;
        } else {
            anyhow::bail!("unsupported runtime file type: {}", source.display());
        }
    }
    Ok(())
}

pub(super) fn copy_dir_recursive(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    copy_directory_contents(source, target)
}

pub(super) fn copy_safe_symbolic_link(
    source: &Path,
    target: &Path,
    source_root: &Path,
) -> Result<()> {
    let link_target = fs::read_link(source)
        .with_context(|| format!("read runtime symbolic link {}", source.display()))?;
    if link_target.is_absolute() {
        anyhow::bail!(
            "runtime symbolic link target must be relative: {}",
            source.display()
        );
    }
    let resolved = fs::canonicalize(
        source
            .parent()
            .ok_or_else(|| anyhow::anyhow!("runtime symbolic link has no parent"))?
            .join(&link_target),
    )
    .with_context(|| format!("resolve runtime symbolic link {}", source.display()))?;
    if !resolved.starts_with(source_root) {
        anyhow::bail!(
            "runtime symbolic link escapes source root: {}",
            source.display()
        );
    }
    create_symbolic_link(&link_target, target, &resolved)
        .with_context(|| format!("copy runtime symbolic link {}", source.display()))
}

pub(super) fn make_executable(path: &Path) -> Result<()> {
    #[cfg(not(unix))]
    let _ = path;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(permissions.mode() | 0o755);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

pub(super) fn make_executable_if_source_is_executable(source: &Path, target: &Path) -> Result<()> {
    #[cfg(not(unix))]
    let _ = (source, target);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let source_mode = fs::metadata(source)?.permissions().mode();
        if source_mode & 0o111 != 0 {
            let mut permissions = fs::metadata(target)?.permissions();
            permissions.set_mode(permissions.mode() | 0o755);
            fs::set_permissions(target, permissions)?;
        }
    }
    Ok(())
}

pub(super) fn write_install_manifest(
    runtime_dir: &Path,
    platform: &str,
    backend: &str,
    catalog: &PrebuiltCatalog,
    entry: &PrebuiltEntry,
    archives: &[DownloadedArchive],
) -> Result<()> {
    let installed_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let primary = archives
        .first()
        .ok_or_else(|| anyhow::anyhow!("prebuilt manifest requires a runtime archive"))?;
    let asset_records = archives
        .iter()
        .map(|archive| {
            json!({
                "role": archive.role,
                "url": archive.url,
                "archive": archive.archive,
                "archive_sha256": archive.sha256,
                "catalog_sha256": archive.catalog_sha256,
            })
        })
        .collect::<Vec<_>>();
    let manifest = json!({
        "schema_version": 3,
        "installed_at": installed_at,
        "platform": platform,
        "backend": backend,
        "source": entry.source,
        "tag": catalog.resolved_tag(entry),
        "url": primary.url,
        "archive_sha256": primary.sha256,
        "catalog_sha256": entry.sha256,
        "archive": entry.archive,
        "launcher": entry.launcher,
        "required_files": entry.required_files,
        "assets": asset_records,
        "submodule_path": catalog.resolved_submodule_path(entry),
        "submodule_commit": catalog.resolved_submodule_commit(entry),
    });
    fs::write(
        runtime_dir.join("prebuilt.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}
