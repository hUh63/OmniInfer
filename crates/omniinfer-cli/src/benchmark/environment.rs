use std::process::Command;

use super::*;

pub(super) fn resolve_device(
    args: &BenchRunArgs,
    backend: &str,
    state: &Value,
) -> Result<(String, String)> {
    let detected_name = detect_device_name(backend, state);
    let device_name = args.device_name.clone().or(detected_name).ok_or_else(|| {
        anyhow::anyhow!("Device name could not be detected. Pass --device-name explicitly.")
    })?;
    let soc = args
        .soc
        .clone()
        .or_else(|| catalog_device_id(&device_name))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Catalog device ID could not be inferred from {device_name:?}. Pass --soc explicitly."
            )
        })?;
    validate_text("device name", &device_name, 256)?;
    validate_text("device SoC", &soc, 256)?;
    Ok((device_name, soc))
}

fn detect_device_name(backend: &str, state: &Value) -> Option<String> {
    if backend.contains("cuda") {
        let visible = json_str(state, "cuda_visible_devices")
            .and_then(|value| value.split(',').next())
            .map(str::trim);
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=index,name", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .filter(|output| output.status.success())?;
        let devices = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                line.split_once(',')
                    .map(|(index, name)| (index.trim().to_string(), name.trim().to_string()))
            })
            .collect::<Vec<_>>();
        return visible
            .and_then(|index| devices.iter().find(|device| device.0 == index))
            .or_else(|| (devices.len() == 1).then(|| &devices[0]))
            .map(|device| device.1.clone());
    }
    if backend.contains("mac") {
        return Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .filter(|value| !value.is_empty());
    }
    if std::env::consts::ARCH == "x86_64" && !backend.contains("rocm") {
        return Some("x86-64 PC CPU".to_string());
    }
    None
}

fn catalog_device_id(device_name: &str) -> Option<String> {
    let normalized = device_name.to_ascii_lowercase();
    if normalized.contains("apple") {
        return Some("apple-silicon".to_string());
    }
    if normalized.contains("radeon 8060s") {
        return Some("radeon-8060s".to_string());
    }
    if normalized.contains("radeon") {
        return Some("amd-radeon-pc".to_string());
    }
    if normalized.contains("x86-64") || normalized.contains("x86_64") {
        return Some("x86-64-cpu".to_string());
    }
    for model in [
        "5090",
        "5080",
        "5070 ti",
        "5070",
        "5060 ti",
        "5060",
        "4090",
        "4080 super",
        "4080",
        "4070 ti super",
        "4070 ti",
        "4070 super",
        "4070",
        "4060 ti",
        "4060",
        "3090 ti",
        "3090",
        "3080 ti",
        "3080",
        "3070 ti",
        "3070",
        "3060 ti",
        "3060",
    ] {
        if normalized.contains(&format!("rtx {model}")) {
            return Some(format!("rtx-{}", model.replace(' ', "-")));
        }
    }
    None
}

pub(super) fn resolve_runtime_provenance(
    args: &BenchRunArgs,
    backend: &str,
    state: &Value,
) -> Result<(String, String)> {
    let manifest = prebuilt_manifest(state, backend);
    let backend_version = args
        .backend_version
        .clone()
        .or_else(|| manifest.as_ref()?.get("tag")?.as_str().map(str::to_string))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Runtime version is unavailable from managed install metadata. Pass --backend-version explicitly."
            )
        })?;
    let build_command = args
        .build_command
        .as_deref()
        .map(str::trim)
        .map(str::to_string)
        .or_else(|| manifest.as_ref().map(|_| format!("omniinfer backend install {backend}")))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Build/install command is unavailable from managed install metadata. Pass --build-command explicitly."
            )
        })?;
    validate_text("runtime version", &backend_version, 256)?;
    validated_command("runtime build command", &build_command)?;
    Ok((backend_version, build_command))
}

fn prebuilt_manifest(state: &Value, backend: &str) -> Option<Value> {
    let runtime_dir = state
        .pointer("/available_backends/data")?
        .as_array()?
        .iter()
        .find(|entry| json_str(entry, "id") == Some(backend))?
        .get("runtime_dir")?
        .as_str()?;
    let raw = fs::read(Path::new(runtime_dir).join("prebuilt.json")).ok()?;
    let manifest: Value = serde_json::from_slice(&raw).ok()?;
    (json_str(&manifest, "backend") == Some(backend)).then_some(manifest)
}
