use super::*;

pub(super) fn detect_wsl_context(requested_distro: Option<&str>) -> Result<WslContext> {
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

pub(super) fn validate_wsl_cuda_gpu(wsl: &WslContext) -> Result<()> {
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

pub(super) fn validate_backend_accelerator(
    backend: &str,
    variant: &PythonRuntimeVariant,
) -> Result<()> {
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

pub(super) fn query_nvidia_driver() -> Result<String> {
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

pub(super) fn require_minimum_driver(actual: &str, minimum: &str) -> Result<()> {
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

pub(super) fn parse_version(value: &str) -> Option<(u32, u32, u32)> {
    let mut parts = value.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

pub(super) fn validate_installed_runtime(
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

pub(super) fn validate_runtime_path(
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

pub(super) fn validate_native_dependencies(
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

pub(super) fn ensure_runtime_not_active(wsl: &WslContext, linux_current: &str) -> Result<()> {
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

pub(super) fn activate_runtime(
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

pub(super) fn rollback_runtime(
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

pub(super) fn write_wsl_file(
    wsl: &WslContext,
    path: &str,
    bytes: &[u8],
    executable: bool,
) -> Result<()> {
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

pub(super) fn write_launcher_manifest(
    runtime_dir: &Path,
    manifest: &LauncherManifest,
) -> Result<()> {
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

pub(super) fn launcher_manifest_matches(runtime_dir: &Path, expected: &LauncherManifest) -> bool {
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

pub(super) fn extract_uv(runtime_dir: &Path, archive: &[u8]) -> Result<PathBuf> {
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

pub(super) fn acquire_install_lock(runtime_dir: &Path) -> Result<InstallLock> {
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

pub(super) fn wsl_path(wsl: &WslContext, windows_path: &Path) -> Result<String> {
    let text = windows_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Windows runtime path is not valid UTF-8"))?;
    run_wsl_text(
        &wsl.executable,
        &wsl.distribution,
        ["wslpath", "-a", "-u", text],
    )
}

pub(super) fn run_wsl_checked<I, S>(
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

pub(super) fn run_wsl_as_checked<I, S>(
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

pub(super) fn run_wsl_as_file_checked<I, S>(
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

pub(super) fn run_wsl<I, S>(wsl: &WslContext, args: I, stdin: Option<&[u8]>) -> Result<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    run_wsl_as(wsl, None, args, stdin)
}

pub(super) fn run_wsl_as<I, S>(
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

pub(super) fn run_wsl_as_file<I, S>(
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

pub(super) fn run_wsl_text<I, S>(executable: &Path, distro: &str, args: I) -> Result<String>
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

pub(super) fn run_command<I, S>(executable: &Path, args: I, stdin: Option<&[u8]>) -> Result<Output>
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

pub(super) fn require_success(output: &Output, action: &str) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }
    anyhow::bail!(
        "{action} failed with {}: {}",
        output.status,
        decode_output(&output.stderr).trim()
    )
}

pub(super) fn decode_output(bytes: &[u8]) -> String {
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

pub(super) fn distro_is_wsl2(name: &str, verbose: &str) -> bool {
    verbose.lines().any(|line| {
        let normalized = line.trim().trim_start_matches('*').trim();
        let distro = normalized.split_whitespace().next();
        let Some(version) = normalized.split_whitespace().last() else {
            return false;
        };
        version == "2" && distro.is_some_and(|distro| distro.eq_ignore_ascii_case(name))
    })
}

pub(super) fn default_distro(verbose: &str) -> Option<String> {
    verbose.lines().find_map(|line| {
        let trimmed = line.trim_start();
        let rest = trimmed.strip_prefix('*')?.trim_start();
        let version = rest.split_whitespace().last()?;
        (version == "2")
            .then(|| rest.split_whitespace().next().map(str::to_string))
            .flatten()
    })
}

pub(super) fn is_internal_distro(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower == "docker-desktop" || lower == "docker-desktop-data" || lower.ends_with("-data")
}

pub(super) fn automount_root_from_c_path(c_root: &str) -> Option<String> {
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

pub(super) fn eligible_distro_names(names: &[String], verbose: &str) -> Vec<String> {
    names
        .iter()
        .filter(|name| !is_internal_distro(name) && distro_is_wsl2(name, verbose))
        .cloned()
        .collect()
}
