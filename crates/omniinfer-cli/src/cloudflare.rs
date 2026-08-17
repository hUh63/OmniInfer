use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use omniinfer_core::paths;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{detach_child_process, hide_child_window};

const CLOUDFLARED_VERSION: &str = "2026.5.0";
const CLOUDFLARED_RELEASE_BASE_URL: &str =
    "https://github.com/cloudflare/cloudflared/releases/download/2026.5.0";
const CLOUDFLARED_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const MAX_CLOUDFLARED_ASSET_BYTES: u64 = 128 * 1024 * 1024;
const MAX_CLOUDFLARED_BINARY_BYTES: u64 = 128 * 1024 * 1024;
const DOWNLOAD_PROGRESS_INTERVAL_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CloudflaredAsset {
    name: &'static str,
    sha256: &'static str,
}

#[derive(Debug, Deserialize, Serialize)]
struct ManagedCloudflaredManifest {
    version: String,
    release: String,
    asset_name: String,
    download_url: String,
    archive_sha256: String,
    binary_sha256: String,
    installed_at_unix: u64,
}

pub(crate) fn resolve_cloudflared(explicit_path: Option<&str>) -> Result<PathBuf> {
    if let Some(path) = explicit_path.filter(|value| !value.trim().is_empty()) {
        let path = PathBuf::from(path);
        if !path.is_file() || !is_executable(&path) {
            anyhow::bail!(
                "cloudflared was not found or is not executable at {}",
                path.display()
            );
        }
        return Ok(path);
    }

    let managed = managed_cloudflared_path();
    let asset =
        cloudflared_asset_for_platform(env::consts::OS, env::consts::ARCH, current_target_abi())?;
    if managed_cloudflared_matches(&managed, asset) {
        return Ok(managed);
    }

    if let Some(path) = find_executable_in_path(cloudflared_executable_name()) {
        return Ok(path);
    }

    println!(
        "cloudflared was not found; downloading pinned release {}...",
        CLOUDFLARED_VERSION
    );
    install_managed_cloudflared(&paths::local_dir(), asset).with_context(|| {
        format!(
            "unable to install the Cloudflare tunnel helper automatically.\n{}",
            cloudflared_install_guidance(asset)
        )
    })
}

fn managed_cloudflared_path() -> PathBuf {
    managed_cloudflared_path_in(&paths::local_dir())
}

fn managed_cloudflared_path_in(local_dir: &Path) -> PathBuf {
    local_dir
        .join("tools")
        .join("cloudflared")
        .join(cloudflared_executable_name())
}

fn managed_cloudflared_manifest_path(local_dir: &Path) -> PathBuf {
    local_dir
        .join("tools")
        .join("cloudflared")
        .join("manifest.json")
}

fn cloudflared_executable_name() -> &'static str {
    if cfg!(windows) {
        "cloudflared.exe"
    } else {
        "cloudflared"
    }
}

fn find_executable_in_path(name: &str) -> Option<PathBuf> {
    let candidate = Path::new(name);
    if candidate.components().count() > 1 && is_executable(candidate) {
        return Some(candidate.to_path_buf());
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|path| is_executable(path))
}

fn current_target_abi() -> &'static str {
    if cfg!(target_abi = "eabihf") {
        "eabihf"
    } else if cfg!(target_abi = "eabi") {
        "eabi"
    } else {
        ""
    }
}

fn cloudflared_asset_for_platform(os: &str, arch: &str, abi: &str) -> Result<CloudflaredAsset> {
    let asset = match (os, arch, abi) {
        ("macos", "aarch64", _) => CloudflaredAsset {
            name: "cloudflared-darwin-arm64.tgz",
            sha256: "116ef11a59fc4f31e7f1bcc4378070cd7ca053fa37b4484b1432bb150b358219",
        },
        ("macos", "x86_64", _) => CloudflaredAsset {
            name: "cloudflared-darwin-amd64.tgz",
            sha256: "7f2c4c8c86e787226804694112682aefacd4cfb98f54508f1a5a841a78bbbef9",
        },
        ("linux", "x86_64", _) => CloudflaredAsset {
            name: "cloudflared-linux-amd64",
            sha256: "0095e46fdc88855d801c4d304cb1f5dd4bd656116c47ab94c2ad0ae7cda1c7ec",
        },
        ("linux", "x86", _) => CloudflaredAsset {
            name: "cloudflared-linux-386",
            sha256: "af63c00d89e92538b40b1e3b8a264558f17c23d706b3b07c1c5a0f21e5f27942",
        },
        ("linux", "aarch64", _) => CloudflaredAsset {
            name: "cloudflared-linux-arm64",
            sha256: "2dc0945345677d27de3ae390a31c3b168866b48766da5f4cfd3fc473ce572303",
        },
        ("linux", "arm", "eabihf") => CloudflaredAsset {
            name: "cloudflared-linux-armhf",
            sha256: "fcd05d6fef48b120c582c26625915bb9bc5713b21105a2c0c142fe72c205adee",
        },
        ("linux", "arm", _) => CloudflaredAsset {
            name: "cloudflared-linux-arm",
            sha256: "22394bc6d820b48a7a273f4d61a8b2f512243404b3f69388fae9632a3d253bb5",
        },
        ("windows", "x86_64", _) => CloudflaredAsset {
            name: "cloudflared-windows-amd64.exe",
            sha256: "f141cded099c239171ad2cea6fb5da0fdaa2bd36104c3074d883f9546519eba7",
        },
        ("windows", "x86", _) => CloudflaredAsset {
            name: "cloudflared-windows-386.exe",
            sha256: "f4294840f044dcfad86d5baccb63d92d3efc3ef1528a6f4962b367477af1dc5f",
        },
        _ => anyhow::bail!("unsupported cloudflared platform: {os} {arch}"),
    };
    Ok(asset)
}

fn cloudflared_asset_url(asset: CloudflaredAsset) -> String {
    format!("{CLOUDFLARED_RELEASE_BASE_URL}/{}", asset.name)
}

fn managed_cloudflared_matches(path: &Path, asset: CloudflaredAsset) -> bool {
    if !path.is_file() || !is_executable(path) {
        return false;
    }
    let manifest_path = path
        .parent()
        .map(|parent| parent.join("manifest.json"))
        .unwrap_or_default();
    let Ok(contents) = fs::read_to_string(manifest_path) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_str::<ManagedCloudflaredManifest>(&contents) else {
        return false;
    };
    if manifest.version != CLOUDFLARED_VERSION
        || manifest.release != CLOUDFLARED_VERSION
        || manifest.asset_name != asset.name
        || !manifest.archive_sha256.eq_ignore_ascii_case(asset.sha256)
    {
        return false;
    }
    sha256_file(path).is_ok_and(|digest| digest.eq_ignore_ascii_case(&manifest.binary_sha256))
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|metadata| metadata.permissions().mode() & 0o111 != 0)
}

#[cfg(windows)]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

fn install_managed_cloudflared(local_dir: &Path, asset: CloudflaredAsset) -> Result<PathBuf> {
    install_managed_cloudflared_with(local_dir, asset, download_cloudflared_asset)
}

fn install_managed_cloudflared_with<F>(
    local_dir: &Path,
    asset: CloudflaredAsset,
    download: F,
) -> Result<PathBuf>
where
    F: FnOnce(&str) -> Result<Vec<u8>>,
{
    let url = cloudflared_asset_url(asset);
    let archive = download(&url)?;
    let archive_sha256 = sha256_hex(&archive);
    if !archive_sha256.eq_ignore_ascii_case(asset.sha256) {
        anyhow::bail!(
            "cloudflared SHA-256 verification failed: expected {}, got {}",
            asset.sha256,
            archive_sha256
        );
    }
    let binary = extract_cloudflared_binary(asset.name, &archive)?;
    let binary_sha256 = sha256_hex(&binary);
    let path = managed_cloudflared_path_in(local_dir);
    write_atomic(&path, &binary, true)?;

    let installed_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let manifest = ManagedCloudflaredManifest {
        version: CLOUDFLARED_VERSION.to_string(),
        release: CLOUDFLARED_VERSION.to_string(),
        asset_name: asset.name.to_string(),
        download_url: url,
        archive_sha256,
        binary_sha256,
        installed_at_unix,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    write_atomic(
        &managed_cloudflared_manifest_path(local_dir),
        &manifest_bytes,
        false,
    )?;
    println!("Installed cloudflared at {}", path.display());
    Ok(path)
}

fn download_cloudflared_asset(url: &str) -> Result<Vec<u8>> {
    let mut last_error = String::new();
    for attempt in 1..=2 {
        match download_cloudflared_asset_once(url) {
            Ok(bytes) => return Ok(bytes),
            Err(error) => {
                last_error = error.to_string();
                if attempt < 2 {
                    println!("cloudflared download failed; retrying once: {last_error}");
                    thread::sleep(Duration::from_secs(attempt));
                }
            }
        }
    }
    anyhow::bail!("failed to download {url} after 2 attempts: {last_error}")
}

fn download_cloudflared_asset_once(url: &str) -> Result<Vec<u8>> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(CLOUDFLARED_DOWNLOAD_TIMEOUT))
        .build()
        .new_agent();
    let mut response = agent
        .get(url)
        .header("Accept", "application/octet-stream")
        .header("User-Agent", "OmniInfer-cloudflared-installer")
        .call()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let total = response.body().content_length();
    if total.is_some_and(|size| size > MAX_CLOUDFLARED_ASSET_BYTES) {
        anyhow::bail!("cloudflared asset exceeds the 128 MiB limit");
    }
    read_bounded_with_progress(
        response.body_mut().as_reader(),
        MAX_CLOUDFLARED_ASSET_BYTES,
        total,
    )
}

fn read_bounded_with_progress(
    mut reader: impl Read,
    limit: u64,
    total: Option<u64>,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(total.unwrap_or_default().min(32 * 1024 * 1024) as usize);
    let mut buffer = [0_u8; 64 * 1024];
    let mut next_report = DOWNLOAD_PROGRESS_INTERVAL_BYTES;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if bytes.len() as u64 + count as u64 > limit {
            anyhow::bail!("cloudflared asset exceeds the 128 MiB limit");
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() as u64 >= next_report {
            print_download_progress(bytes.len() as u64, total);
            next_report = bytes.len() as u64 + DOWNLOAD_PROGRESS_INTERVAL_BYTES;
        }
    }
    print_download_progress(bytes.len() as u64, total);
    Ok(bytes)
}

fn print_download_progress(downloaded: u64, total: Option<u64>) {
    let downloaded_mib = downloaded as f64 / (1024.0 * 1024.0);
    if let Some(total) = total {
        let total_mib = total as f64 / (1024.0 * 1024.0);
        println!("Downloading cloudflared: {downloaded_mib:.1}/{total_mib:.1} MiB");
    } else {
        println!("Downloading cloudflared: {downloaded_mib:.1} MiB");
    }
}

fn extract_cloudflared_binary(asset_name: &str, archive: &[u8]) -> Result<Vec<u8>> {
    if !asset_name.ends_with(".tgz") {
        if archive.len() as u64 > MAX_CLOUDFLARED_BINARY_BYTES {
            anyhow::bail!("cloudflared binary exceeds the 128 MiB limit");
        }
        return Ok(archive.to_vec());
    }

    let decoder = GzDecoder::new(archive);
    let mut tar = tar::Archive::new(decoder);
    let mut binary = None;
    for entry in tar.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?;
        if path.file_name().and_then(|name| name.to_str()) != Some("cloudflared") {
            continue;
        }
        if binary.is_some() {
            anyhow::bail!("cloudflared archive contains multiple cloudflared binaries");
        }
        binary = Some(read_bounded(
            &mut entry,
            MAX_CLOUDFLARED_BINARY_BYTES,
            "cloudflared binary",
        )?);
    }
    binary.ok_or_else(|| {
        anyhow::anyhow!("cloudflared archive did not contain a regular cloudflared binary")
    })
}

fn read_bounded(mut reader: impl Read, limit: u64, label: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if bytes.len() as u64 + count as u64 > limit {
            anyhow::bail!("{label} exceeds the {} MiB limit", limit / 1024 / 1024);
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    Ok(bytes)
}

fn write_atomic(path: &Path, bytes: &[u8], executable: bool) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("managed cloudflared path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temp = unique_sibling_path(path, "tmp");
    let prepare_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        set_executable(&temp, executable)
    })();
    if let Err(error) = prepare_result {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }

    let backup = unique_sibling_path(path, "backup");
    let had_existing = path.exists();
    if had_existing {
        fs::rename(path, &backup)?;
    }
    if let Err(error) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        if had_existing {
            let _ = fs::rename(&backup, path);
        }
        return Err(error.into());
    }
    if had_existing {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

fn unique_sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("cloudflared");
    path.with_file_name(format!(".{name}-{}-{nonce}.{suffix}", std::process::id()))
}

#[cfg(unix)]
fn set_executable(path: &Path, executable: bool) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = if executable { 0o755 } else { 0o644 };
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(windows)]
fn set_executable(_path: &Path, _executable: bool) -> Result<()> {
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let bytes = read_bounded(
        file,
        MAX_CLOUDFLARED_BINARY_BYTES,
        "managed cloudflared binary",
    )?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn cloudflared_install_guidance(asset: CloudflaredAsset) -> String {
    let platform_command = match env::consts::OS {
        "macos" => "  brew install cloudflared",
        "windows" => "  winget install --id Cloudflare.cloudflared",
        _ => {
            "  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        }
    };
    format!(
        "Install cloudflared manually:\n{platform_command}\nThen retry, or pass --cloudflared-path with the installed executable.\nPinned asset: {}\nExpected SHA-256: {}",
        cloudflared_asset_url(asset),
        asset.sha256
    )
}

pub(crate) fn start_cloudflare_quick_tunnel(
    cloudflared: &Path,
    local_url: &str,
    log_path: &Path,
    detach: bool,
    mut on_spawn: impl FnMut(&std::process::Child) -> Result<()>,
) -> Result<(std::process::Child, String)> {
    let output = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let initial_log_length = output.metadata()?.len();
    let stdout = output.try_clone()?;
    let stderr = output;
    let mut reader = BufReader::new(OpenOptions::new().read(true).open(log_path)?);
    reader.seek(SeekFrom::Start(initial_log_length))?;
    let mut command = ProcessCommand::new(cloudflared);
    command
        .args(["tunnel", "--url", local_url])
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    hide_child_window(&mut command);
    if detach {
        detach_child_process(&mut command);
    }
    let mut child = command.spawn()?;
    if let Err(error) = on_spawn(&child) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut tail = Vec::new();
    let mut public_url = None;
    while Instant::now() < deadline {
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim_end().to_string();
            if tail.len() == 10 {
                tail.remove(0);
            }
            tail.push(line.clone());
            if public_url.is_none()
                && let Some(url) = parse_trycloudflare_url(&line)
            {
                public_url = Some((url, Instant::now()));
            }
        }
        if let Some(status) = child.try_wait()? {
            anyhow::bail!(
                "cloudflared exited before creating a Quick Tunnel with status {status}.{}",
                format_log_tail(&tail)
            );
        }
        if let Some((url, observed_at)) = public_url.as_ref()
            && observed_at.elapsed() >= Duration::from_millis(200)
        {
            return Ok((child, url.clone()));
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
    anyhow::bail!(
        "Timed out waiting for Cloudflare Quick Tunnel URL.{}",
        format_log_tail(&tail)
    )
}

fn parse_trycloudflare_url(line: &str) -> Option<String> {
    line.split(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | '(' | ')' | '[' | ']'))
        .find(|part| part.starts_with("https://") && part.contains(".trycloudflare.com"))
        .map(|part| {
            part.trim_end_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '/')
                .trim_end_matches('/')
                .to_string()
        })
}

fn format_log_tail(lines: &[String]) -> String {
    if lines.is_empty() {
        String::new()
    } else {
        format!("\ncloudflared log tail:\n{}", lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;

    #[test]
    fn selects_macos_arm64_asset() {
        let asset = cloudflared_asset_for_platform("macos", "aarch64", "").expect("asset");
        assert_eq!(asset.name, "cloudflared-darwin-arm64.tgz");
        assert_eq!(
            asset.sha256,
            "116ef11a59fc4f31e7f1bcc4378070cd7ca053fa37b4484b1432bb150b358219"
        );
    }

    #[test]
    fn selects_linux_arm_asset_by_abi() {
        assert_eq!(
            cloudflared_asset_for_platform("linux", "arm", "eabihf")
                .expect("armhf asset")
                .name,
            "cloudflared-linux-armhf"
        );
        assert_eq!(
            cloudflared_asset_for_platform("linux", "arm", "eabi")
                .expect("arm asset")
                .name,
            "cloudflared-linux-arm"
        );
    }

    #[test]
    fn rejects_unsupported_platform() {
        let error = cloudflared_asset_for_platform("plan9", "mips", "")
            .expect_err("unsupported platform should fail");
        assert!(
            error
                .to_string()
                .contains("unsupported cloudflared platform")
        );
    }

    #[test]
    fn extracts_regular_binary_from_tgz() {
        let payload = b"test-cloudflared";
        let archive = cloudflared_tgz(payload);
        assert_eq!(
            extract_cloudflared_binary("cloudflared-darwin-arm64.tgz", &archive)
                .expect("extract binary"),
            payload
        );
    }

    #[test]
    fn rejects_download_with_wrong_digest() {
        let root = test_local_dir("digest-mismatch");
        let asset = CloudflaredAsset {
            name: "cloudflared-test",
            sha256: "00",
        };
        let error = install_managed_cloudflared_with(&root, asset, |_| Ok(b"wrong".to_vec()))
            .expect_err("digest mismatch should fail");
        assert!(error.to_string().contains("SHA-256 verification failed"));
        assert!(!managed_cloudflared_path_in(&root).exists());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn installs_and_validates_managed_binary() {
        let root = test_local_dir("managed-install");
        let payload = b"test-cloudflared-binary";
        let digest = Box::leak(sha256_hex(payload).into_boxed_str());
        let asset = CloudflaredAsset {
            name: "cloudflared-test",
            sha256: digest,
        };
        let path = install_managed_cloudflared_with(&root, asset, |_| Ok(payload.to_vec()))
            .expect("install");
        assert_eq!(fs::read(&path).expect("read binary"), payload);
        assert!(managed_cloudflared_matches(&path, asset));

        fs::write(&path, b"tampered").expect("tamper binary");
        assert!(!managed_cloudflared_matches(&path, asset));
        fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_helper_that_exits_after_printing_tunnel_url() {
        use std::os::unix::fs::PermissionsExt;

        let root = test_local_dir("exited-helper");
        fs::create_dir_all(&root).expect("create test directory");
        let helper = root.join("cloudflared");
        let log_path = root.join("cloudflared.log");
        fs::write(
            &helper,
            "#!/bin/sh\n\
             echo 'https://exited-helper.trycloudflare.com'\n\
             echo 'fatal: tunnel registration failed' >&2\n\
             exit 9\n",
        )
        .expect("write fake helper");
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o755))
            .expect("make fake helper executable");

        let error = start_cloudflare_quick_tunnel(
            &helper,
            "http://127.0.0.1:8080",
            &log_path,
            false,
            |_| Ok(()),
        )
        .expect_err("exited helper must not be accepted");
        let message = error.to_string();
        assert!(message.contains("status exit status: 9"), "{message}");
        assert!(
            message.contains("fatal: tunnel registration failed"),
            "{message}"
        );
        fs::remove_dir_all(root).ok();
    }

    fn cloudflared_tgz(payload: &[u8]) -> Vec<u8> {
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(payload.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        archive
            .append_data(&mut header, "cloudflared", payload)
            .expect("append cloudflared");
        let encoder = archive.into_inner().expect("finish tar");
        encoder.finish().expect("finish gzip")
    }

    fn test_local_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        env::temp_dir().join(format!(
            "omniinfer-cloudflared-{name}-{}-{nonce}",
            std::process::id()
        ))
    }
}
