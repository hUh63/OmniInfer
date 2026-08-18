use super::*;

pub(super) fn extract_archive(bytes: &[u8], archive_type: &str, destination: &Path) -> Result<()> {
    match archive_type.to_ascii_lowercase().as_str() {
        "tar.gz" | "tgz" => extract_tar_archive(bytes, destination),
        "zip" => {
            let reader = Cursor::new(bytes);
            let mut archive = zip::ZipArchive::new(reader)?;
            for index in 0..archive.len() {
                let mut file = archive.by_index(index)?;
                let enclosed = file
                    .enclosed_name()
                    .ok_or_else(|| anyhow::anyhow!("unsafe zip path: {}", file.name()))?
                    .to_path_buf();
                validate_archive_path(&enclosed)?;
                let target = destination.join(&enclosed);
                if file.is_dir() {
                    fs::create_dir_all(&target)?;
                } else {
                    if let Some(parent) = target.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    let mut out = File::create(&target)?;
                    std::io::copy(&mut file, &mut out)?;
                    #[cfg(unix)]
                    if let Some(mode) = file.unix_mode() {
                        use std::os::unix::fs::PermissionsExt;
                        fs::set_permissions(&target, fs::Permissions::from_mode(mode))?;
                    }
                }
            }
            Ok(())
        }
        other => anyhow::bail!("unsupported prebuilt archive type: {other}"),
    }
}

pub(super) fn extract_tar_archive(bytes: &[u8], destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    let destination_root = fs::canonicalize(destination)
        .with_context(|| format!("resolve tar destination {}", destination.display()))?;
    let links = inspect_tar_entries(bytes)?;

    let decoder = GzDecoder::new(Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let entry_type = entry.header().entry_type();
        if (entry_type.is_file() || entry_type.is_dir()) && !entry.unpack_in(&destination_root)? {
            anyhow::bail!("unsafe tar path: {}", path.display());
        }
    }

    create_tar_links(&destination_root, &links)
}

pub(super) fn inspect_tar_entries(bytes: &[u8]) -> Result<Vec<TarLink>> {
    let decoder = GzDecoder::new(Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    let mut links = Vec::new();
    for entry in archive.entries()? {
        let entry = entry?;
        let path = entry.path()?.to_path_buf();
        validate_archive_path(&path)?;
        let entry_type = entry.header().entry_type();
        if entry_type.is_file() || entry_type.is_dir() {
            continue;
        }
        let kind = if entry_type.is_symlink() {
            TarLinkKind::Symbolic
        } else if entry_type.is_hard_link() {
            TarLinkKind::Hard
        } else {
            anyhow::bail!("unsupported tar entry type for {}", path.display());
        };
        let target = entry
            .link_name()?
            .ok_or_else(|| anyhow::anyhow!("tar link {} has no target", path.display()))?
            .into_owned();
        let resolved_target = resolve_tar_link_target(&path, &target, kind)?;
        links.push(TarLink {
            path,
            target,
            resolved_target,
            kind,
        });
    }
    validate_tar_link_graph(&links)?;
    Ok(links)
}

pub(super) fn resolve_tar_link_target(
    path: &Path,
    target: &Path,
    kind: TarLinkKind,
) -> Result<PathBuf> {
    let mut resolved = PathBuf::new();
    if kind == TarLinkKind::Symbolic
        && let Some(parent) = path.parent()
    {
        resolved.push(parent);
    }
    for component in target.components() {
        match component {
            Component::Normal(value) => resolved.push(value),
            Component::CurDir => {}
            Component::ParentDir => {
                if !resolved.pop() {
                    anyhow::bail!(
                        "tar link target escapes staging root: {} -> {}",
                        path.display(),
                        target.display()
                    );
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!(
                    "tar link target must be relative: {} -> {}",
                    path.display(),
                    target.display()
                );
            }
        }
    }
    if resolved.as_os_str().is_empty() {
        anyhow::bail!(
            "tar link target is empty: {} -> {}",
            path.display(),
            target.display()
        );
    }
    Ok(resolved)
}

pub(super) fn validate_tar_link_graph(links: &[TarLink]) -> Result<()> {
    let symbolic_links = links
        .iter()
        .filter(|link| link.kind == TarLinkKind::Symbolic)
        .map(|link| (link.path.clone(), link.resolved_target.clone()))
        .collect::<HashMap<_, _>>();
    if symbolic_links.len()
        != links
            .iter()
            .filter(|link| link.kind == TarLinkKind::Symbolic)
            .count()
    {
        anyhow::bail!("tar archive contains duplicate symbolic link paths");
    }

    for link in links {
        let mut ancestor = link.path.parent();
        while let Some(path) = ancestor {
            if symbolic_links.contains_key(path) {
                anyhow::bail!(
                    "tar link path traverses another symbolic link: {}",
                    link.path.display()
                );
            }
            ancestor = path.parent();
        }

        let mut current = link.resolved_target.as_path();
        let mut visited = HashSet::new();
        while let Some(next) = symbolic_links.get(current) {
            if !visited.insert(current.to_path_buf()) {
                anyhow::bail!("tar symbolic link cycle at {}", link.path.display());
            }
            current = next;
        }
    }
    Ok(())
}

pub(super) fn create_tar_links(destination_root: &Path, links: &[TarLink]) -> Result<()> {
    let symbolic_targets = links
        .iter()
        .filter(|link| link.kind == TarLinkKind::Symbolic)
        .map(|link| (link.path.clone(), link.resolved_target.clone()))
        .collect::<HashMap<_, _>>();

    for link in links.iter().filter(|link| link.kind == TarLinkKind::Hard) {
        let final_target = resolve_final_tar_target(&link.resolved_target, &symbolic_targets)?;
        let source = canonical_target_in_root(destination_root, &final_target, link)?;
        if !source.is_file() {
            anyhow::bail!("tar hard link target is not a file: {}", source.display());
        }
        let target = checked_link_destination(destination_root, &link.path)?;
        fs::hard_link(&source, &target).with_context(|| {
            format!(
                "create tar hard link {} -> {}",
                target.display(),
                source.display()
            )
        })?;
    }

    for link in links
        .iter()
        .filter(|link| link.kind == TarLinkKind::Symbolic)
    {
        let final_target = resolve_final_tar_target(&link.resolved_target, &symbolic_targets)?;
        let source = canonical_target_in_root(destination_root, &final_target, link)?;
        let target = checked_link_destination(destination_root, &link.path)?;
        create_symbolic_link(&link.target, &target, &source)?;
    }

    for link in links
        .iter()
        .filter(|link| link.kind == TarLinkKind::Symbolic)
    {
        let target = destination_root.join(&link.path);
        let resolved = fs::canonicalize(&target)
            .with_context(|| format!("resolve extracted tar link {}", target.display()))?;
        if !resolved.starts_with(destination_root) {
            anyhow::bail!(
                "extracted tar link escapes staging root: {}",
                target.display()
            );
        }
    }
    Ok(())
}

pub(super) fn resolve_final_tar_target(
    target: &Path,
    symbolic_targets: &HashMap<PathBuf, PathBuf>,
) -> Result<PathBuf> {
    let mut current = target.to_path_buf();
    let mut visited = HashSet::new();
    while let Some(next) = symbolic_targets.get(&current) {
        if !visited.insert(current.clone()) {
            anyhow::bail!("tar symbolic link cycle at {}", current.display());
        }
        current = next.clone();
    }
    Ok(current)
}

pub(super) fn canonical_target_in_root(
    destination_root: &Path,
    relative_target: &Path,
    link: &TarLink,
) -> Result<PathBuf> {
    let target = destination_root.join(relative_target);
    let canonical = fs::canonicalize(&target).with_context(|| {
        format!(
            "tar link target does not exist: {} -> {}",
            link.path.display(),
            link.target.display()
        )
    })?;
    if !canonical.starts_with(destination_root) {
        anyhow::bail!(
            "tar link target escapes staging root: {} -> {}",
            link.path.display(),
            link.target.display()
        );
    }
    Ok(canonical)
}

pub(super) fn checked_link_destination(
    destination_root: &Path,
    relative: &Path,
) -> Result<PathBuf> {
    let target = destination_root.join(relative);
    let parent = target
        .parent()
        .ok_or_else(|| anyhow::anyhow!("tar link has no parent: {}", relative.display()))?;
    fs::create_dir_all(parent)?;
    let canonical_parent = fs::canonicalize(parent)
        .with_context(|| format!("resolve tar link parent {}", parent.display()))?;
    if !canonical_parent.starts_with(destination_root) {
        anyhow::bail!("tar link path escapes staging root: {}", relative.display());
    }
    if fs::symlink_metadata(&target).is_ok() {
        anyhow::bail!("tar link path already exists: {}", relative.display());
    }
    Ok(target)
}

pub(super) fn create_symbolic_link(
    target: &Path,
    link: &Path,
    resolved_target: &Path,
) -> Result<()> {
    #[cfg(unix)]
    {
        let _ = resolved_target;
        std::os::unix::fs::symlink(target, link)?;
    }
    #[cfg(windows)]
    {
        if resolved_target.is_dir() {
            std::os::windows::fs::symlink_dir(target, link)?;
        } else {
            std::os::windows::fs::symlink_file(target, link)?;
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, link, resolved_target);
        anyhow::bail!("symbolic links are not supported on this platform");
    }
    Ok(())
}

pub(super) fn validate_archive_path(path: &Path) -> Result<()> {
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            _ => anyhow::bail!("unsafe archive path: {}", path.display()),
        }
    }
    Ok(())
}
