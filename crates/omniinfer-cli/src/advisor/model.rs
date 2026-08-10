use super::*;
use std::io::{BufReader, Read, Seek, SeekFrom};

const QUANT_PATTERNS: &[&str] = &[
    "UD-Q8_K_XL",
    "UD-Q8_K_L",
    "UD-Q8_K_M",
    "UD-Q8_K_S",
    "UD-Q6_K_XL",
    "UD-Q6_K_L",
    "UD-Q6_K_M",
    "UD-Q6_K_S",
    "UD-Q5_K_XL",
    "UD-Q5_K_L",
    "UD-Q5_K_M",
    "UD-Q5_K_S",
    "UD-Q4_K_XL",
    "UD-Q4_K_L",
    "UD-Q4_K_M",
    "UD-Q4_K_S",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q4_K_M",
    "Q4_0",
    "Q3_K_M",
    "Q2_K",
    "BF16",
    "F16",
    "F32",
];

pub fn inspect_payload(model: &str, mmproj: Option<&str>, ctx_size: Option<u32>) -> Result<Value> {
    let (resolved_model, model_path, resolved_mmproj, artifact_kind, warnings) =
        resolve_model_artifact(model, mmproj)?;
    let size_gib = model_path.as_deref().and_then(path_size_gib);
    let mmproj_size_gib = resolved_mmproj.as_deref().and_then(path_size_gib);
    let quantization = infer_quantization(&resolved_model);
    let params_b = infer_params_b(&resolved_model);
    let format = model_format(&artifact_kind, model_path.as_deref(), &resolved_model);
    let vla_architecture = model_path
        .as_deref()
        .filter(|path| path.is_file())
        .and_then(detect_vla_architecture);
    let capabilities = infer_model_capabilities(
        &resolved_model,
        resolved_mmproj.as_deref(),
        vla_architecture.as_deref(),
    );
    Ok(json!({
        "object": "advisor.model",
        "input": model,
        "model": resolved_model,
        "model_path": model_path.as_ref().map(|path| path.display().to_string()),
        "mmproj": resolved_mmproj.as_ref().map(|path| path.display().to_string()),
        "format": format,
        "artifact_kind": artifact_kind,
        "exists": model_path.as_deref().map(Path::exists).unwrap_or(false),
        "size_gib": size_gib,
        "mmproj_size_gib": mmproj_size_gib,
        "quantization": quantization,
        "params_b": params_b,
        "capabilities": capabilities,
        "vla_architecture": vla_architecture,
        "estimate": memory_estimate(size_gib, mmproj_size_gib, params_b, ctx_size.unwrap_or(DEFAULT_CONTEXT_SIZE)),
        "warnings": warnings,
    }))
}

fn resolve_model_artifact(
    model: &str,
    mmproj: Option<&str>,
) -> Result<(String, Option<PathBuf>, Option<PathBuf>, String, Vec<Value>)> {
    let text = model.trim();
    if text.is_empty() {
        anyhow::bail!("model reference must not be empty");
    }
    let path = expand_path(text);
    let resolved_mmproj = mmproj
        .filter(|value| !value.trim().is_empty())
        .map(expand_path)
        .map(|path| path.canonicalize().unwrap_or(path));
    let mut warnings = Vec::new();
    if path.exists() {
        let path = path.canonicalize().unwrap_or(path);
        if path.is_dir() {
            return Ok((
                path.display().to_string(),
                Some(path),
                resolved_mmproj,
                "directory".to_string(),
                warnings,
            ));
        }
        return Ok((
            path.display().to_string(),
            Some(path),
            resolved_mmproj,
            "file".to_string(),
            warnings,
        ));
    }
    if (text.contains('/') || text.contains(':')) && !text.to_ascii_lowercase().ends_with(".gguf") {
        return Ok((
            text.to_string(),
            None,
            resolved_mmproj,
            "reference".to_string(),
            warnings,
        ));
    }
    warnings.push(Value::String(format!(
        "model path does not exist locally: {}",
        path.display()
    )));
    Ok((
        path.display().to_string(),
        Some(path),
        resolved_mmproj,
        "missing".to_string(),
        warnings,
    ))
}

fn expand_path(value: &str) -> PathBuf {
    let text = value.trim();
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    let path = PathBuf::from(text);
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn model_format(artifact_kind: &str, model_path: Option<&Path>, model_ref: &str) -> String {
    if artifact_kind == "reference" {
        return "hf-reference".to_string();
    }
    if model_path.is_some_and(Path::is_dir) {
        return "directory".to_string();
    }
    let extension = model_path
        .and_then(Path::extension)
        .and_then(|value| value.to_str())
        .or_else(|| {
            Path::new(model_ref)
                .extension()
                .and_then(|value| value.to_str())
        });
    if extension.is_some_and(|value| value.eq_ignore_ascii_case("gguf")) {
        return "gguf".to_string();
    }
    if extension.is_some_and(|value| value.eq_ignore_ascii_case("safetensors")) {
        return "safetensors".to_string();
    }
    if artifact_kind == "directory" {
        return "directory".to_string();
    }
    "unknown".to_string()
}

fn path_size_gib(path: &Path) -> Option<f64> {
    path.is_file()
        .then(|| path.metadata().ok())
        .flatten()
        .map(|metadata| round_gib(metadata.len() as f64 / 1024.0 / 1024.0 / 1024.0))
}

fn infer_quantization(text: &str) -> Option<String> {
    let upper = Path::new(text)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(text)
        .to_ascii_uppercase();
    QUANT_PATTERNS
        .iter()
        .find(|quant| upper.contains(**quant))
        .map(|value| (*value).to_string())
}

fn infer_params_b(text: &str) -> Option<f64> {
    let name = Path::new(text)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(text)
        .as_bytes();
    let mut best = None;
    let mut index = 0;
    while index < name.len() {
        if !name[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < name.len() && (name[index].is_ascii_digit() || name[index] == b'.') {
            index += 1;
        }
        if index >= name.len() || !matches!(name[index], b'B' | b'b' | b'M' | b'm') {
            continue;
        }
        let Ok(mut value) = std::str::from_utf8(&name[start..index])
            .unwrap_or("")
            .parse::<f64>()
        else {
            continue;
        };
        if matches!(name[index], b'M' | b'm') {
            value /= 1000.0;
        }
        best = Some(best.map_or(value, |current: f64| current.max(value)));
        index += 1;
    }
    best
}

fn infer_model_capabilities(
    text: &str,
    mmproj_path: Option<&Path>,
    vla_architecture: Option<&str>,
) -> Vec<String> {
    let lower = Path::new(text)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(text)
        .to_ascii_lowercase();
    let mut capabilities = if vla_architecture.is_some() {
        vec![
            "action".to_string(),
            "robotics".to_string(),
            "vision".to_string(),
        ]
    } else {
        vec!["chat".to_string()]
    };
    if mmproj_path.is_some()
        || ["vl", "vision", "mmproj", "multimodal"]
            .iter()
            .any(|token| lower.contains(token))
    {
        capabilities.push("vision".to_string());
    }
    if ["embed", "embedding", "bge", "nomic"]
        .iter()
        .any(|token| lower.contains(token))
    {
        capabilities.push("embedding".to_string());
    }
    capabilities.sort();
    capabilities.dedup();
    capabilities
}

const VLA_ARCHITECTURE_KEYS: &[&str] = &[
    "general.architecture",
    "smolvla.architecture",
    "pi0.architecture",
    "pi05.architecture",
    "evo1.architecture",
    "gr00t_n1_5.architecture",
    "gr00t_n1_6.architecture",
    "gr00t_n1_7.architecture",
    "bitvla.architecture",
    "openvla_oft.architecture",
    "vla_jepa.architecture",
    "vla_adapter.architecture",
];

const VLA_ARCHITECTURES: &[&str] = &[
    "smolvla",
    "pi0",
    "pi05",
    "evo1",
    "gr00t_n1_5",
    "gr00t_n1_6",
    "gr00t_n1_7",
    "bitvla",
    "openvla_oft",
    "vla_jepa",
    "vla_adapter",
];

fn detect_vla_architecture(path: &Path) -> Option<String> {
    match path.extension().and_then(|value| value.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("gguf") => {
            detect_gguf_vla_architecture(path)
        }
        Some(extension) if extension.eq_ignore_ascii_case("safetensors") => {
            detect_safetensors_vla_architecture(path)
        }
        _ => None,
    }
}

fn detect_gguf_vla_architecture(path: &Path) -> Option<String> {
    let mut reader = BufReader::new(std::fs::File::open(path).ok()?);
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut reader).ok()?;
    if !matches!(version, 2 | 3) {
        return None;
    }
    let _tensor_count = read_u64(&mut reader).ok()?;
    let metadata_count = read_u64(&mut reader).ok()?;
    if metadata_count > 1_000_000 {
        return None;
    }
    for _ in 0..metadata_count {
        let key = read_gguf_string(&mut reader, 1 << 20).ok()?;
        let value_type = read_u32(&mut reader).ok()?;
        if VLA_ARCHITECTURE_KEYS.contains(&key.as_str()) && value_type == 8 {
            let architecture = read_gguf_string(&mut reader, 1 << 20).ok()?;
            if VLA_ARCHITECTURES.contains(&architecture.as_str()) {
                return Some(architecture);
            }
        } else {
            skip_gguf_value(&mut reader, value_type, 0).ok()?;
        }
    }
    None
}

fn detect_safetensors_vla_architecture(path: &Path) -> Option<String> {
    let mut reader = BufReader::new(std::fs::File::open(path).ok()?);
    let header_size = read_u64(&mut reader).ok()?;
    if header_size == 0 || header_size > (1 << 28) {
        return None;
    }
    let patterns = [
        b"vlm_with_expert.vlm.".as_slice(),
        b"paligemma_with_expert.paligemma.".as_slice(),
        b"action_time_mlp_in".as_slice(),
        b"time_mlp_in".as_slice(),
    ];
    let mut found = [false; 4];
    let max_pattern = patterns.iter().map(|pattern| pattern.len()).max()?;
    let mut carry = Vec::new();
    let mut remaining = header_size;
    let mut chunk = vec![0_u8; 64 * 1024];
    while remaining > 0 {
        let wanted = usize::try_from(remaining.min(chunk.len() as u64)).ok()?;
        reader.read_exact(&mut chunk[..wanted]).ok()?;
        let mut window = Vec::with_capacity(carry.len() + wanted);
        window.extend_from_slice(&carry);
        window.extend_from_slice(&chunk[..wanted]);
        for (index, pattern) in patterns.iter().enumerate() {
            found[index] |= window
                .windows(pattern.len())
                .any(|candidate| candidate == *pattern);
        }
        let keep = usize::min(max_pattern.saturating_sub(1), window.len());
        carry.clear();
        carry.extend_from_slice(&window[window.len() - keep..]);
        remaining -= wanted as u64;
    }
    if found[0] {
        Some("smolvla".to_string())
    } else if found[1] && found[2] {
        Some("pi0".to_string())
    } else if found[1] && found[3] {
        Some("pi05".to_string())
    } else {
        None
    }
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_gguf_string(reader: &mut impl Read, maximum: u64) -> std::io::Result<String> {
    let length = read_u64(reader)?;
    if length > maximum {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "GGUF string exceeds inspection limit",
        ));
    }
    let mut bytes = vec![0_u8; length as usize];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

fn skip_gguf_value(
    reader: &mut (impl Read + Seek),
    value_type: u32,
    depth: u8,
) -> std::io::Result<()> {
    let fixed_size = match value_type {
        0 | 1 | 7 => Some(1_u64),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    };
    if let Some(size) = fixed_size {
        return skip_bytes(reader, size);
    }
    match value_type {
        8 => {
            let length = read_u64(reader)?;
            skip_bytes(reader, length)
        }
        9 if depth == 0 => {
            let element_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > 10_000_000 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "GGUF array exceeds inspection limit",
                ));
            }
            let element_size = match element_type {
                0 | 1 | 7 => Some(1_u64),
                2 | 3 => Some(2),
                4..=6 => Some(4),
                10..=12 => Some(8),
                _ => None,
            };
            if let Some(size) = element_size {
                return skip_bytes(
                    reader,
                    count.checked_mul(size).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "GGUF array size overflow",
                        )
                    })?,
                );
            }
            for _ in 0..count {
                skip_gguf_value(reader, element_type, depth + 1)?;
            }
            Ok(())
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported GGUF metadata value type",
        )),
    }
}

fn skip_bytes(reader: &mut impl Seek, length: u64) -> std::io::Result<()> {
    let offset = i64::try_from(length).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "GGUF metadata offset exceeds inspection limit",
        )
    })?;
    reader.seek(SeekFrom::Current(offset))?;
    Ok(())
}

pub(super) fn memory_estimate(
    size_gib: Option<f64>,
    mmproj_size_gib: Option<f64>,
    params_b: Option<f64>,
    ctx_size: u32,
) -> Value {
    let Some(size_gib) = size_gib else {
        return json!({
            "estimated_gpu_memory_gib": null,
            "estimated_ram_gib": null,
            "estimated_kv_cache_gib": null,
            "breakdown": unknown_memory_breakdown(ctx_size),
            "estimate_source": "unknown",
            "confidence": "low",
            "notes": ["local model size is unknown; fit cannot be estimated safely"],
        });
    };
    let base = size_gib + mmproj_size_gib.unwrap_or(0.0);
    let ctx_factor = f64::from(ctx_size.max(1)) / f64::from(DEFAULT_CONTEXT_SIZE);
    let param_factor = params_b.unwrap_or_else(|| f64::max(base * 2.0, 1.0));
    let weights = round_gib(size_gib);
    let mmproj = round_gib(mmproj_size_gib.unwrap_or(0.0));
    let kv_cache = round_gib(f64::max(0.25, param_factor * 0.03 * ctx_factor));
    let activation = round_gib(f64::max(0.12, param_factor * 0.01 * ctx_factor.min(4.0)));
    let framework_overhead = round_gib(f64::max(0.35, base * 0.08));
    let allocator_slack = round_gib(f64::max(0.15, base * 0.04));
    let runtime_overhead = round_gib(framework_overhead + allocator_slack);
    let required = round_gib(weights + mmproj + kv_cache + activation + runtime_overhead);
    let confidence = if params_b.is_some() { "medium" } else { "low" };
    let breakdown = json!({
        "weights_gib": weights,
        "mmproj_gib": mmproj,
        "kv_cache_gib": kv_cache,
        "activation_gib": activation,
        "framework_overhead_gib": framework_overhead,
        "allocator_slack_gib": allocator_slack,
        "runtime_overhead_gib": runtime_overhead,
        "total_gib": required,
        "context_size": ctx_size,
        "assumptions": [
            "weights are approximated from local artifact file size",
            "KV cache is estimated from inferred parameter count and requested context",
            "activation and framework overhead include conservative runtime buffers and allocator slack"
        ],
    });
    json!({
        "estimated_gpu_memory_gib": required,
        "estimated_ram_gib": required,
        "estimated_kv_cache_gib": kv_cache,
        "weight_and_projector_gib": round_gib(weights + mmproj),
        "activation_gib": activation,
        "framework_overhead_gib": framework_overhead,
        "allocator_slack_gib": allocator_slack,
        "overhead_gib": runtime_overhead,
        "breakdown": breakdown,
        "context_size": ctx_size,
        "estimate_source": "file_size_heuristic",
        "confidence": confidence,
        "notes": ["Estimate uses local file size plus KV cache, activation, framework overhead, and allocator slack; backend logs or benchmark results are authoritative."],
    })
}

fn unknown_memory_breakdown(ctx_size: u32) -> Value {
    json!({
        "weights_gib": null,
        "mmproj_gib": null,
        "kv_cache_gib": null,
        "activation_gib": null,
        "framework_overhead_gib": null,
        "allocator_slack_gib": null,
        "runtime_overhead_gib": null,
        "total_gib": null,
        "context_size": ctx_size,
        "assumptions": ["local model size is unknown"],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn identifies_vla_gguf_architecture_and_capabilities() {
        let path = temp_file("advisor-vla-gguf", "gguf");
        write_test_gguf(
            &path,
            &[
                ("general.name", "SmolVLA test"),
                ("smolvla.architecture", "smolvla"),
            ],
        );

        let payload = inspect_payload(path.to_str().unwrap(), None, Some(4096)).unwrap();
        assert_eq!(payload["format"], "gguf");
        assert_eq!(payload["vla_architecture"], "smolvla");
        assert!(model_has_capability(&payload, "action"));
        assert!(model_has_capability(&payload, "vision"));
        assert!(!model_has_capability(&payload, "chat"));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn identifies_supported_vla_safetensors_namespaces() {
        let smolvla = temp_file("advisor-vla-smolvla", "safetensors");
        write_test_safetensors(&smolvla, br#"{"vlm_with_expert.vlm.layer": {}}"#);
        assert_eq!(
            detect_vla_architecture(&smolvla).as_deref(),
            Some("smolvla")
        );

        let pi05 = temp_file("advisor-vla-pi05", "safetensors");
        write_test_safetensors(
            &pi05,
            br#"{"paligemma_with_expert.paligemma.layer": {}, "time_mlp_in": {}}"#,
        );
        assert_eq!(detect_vla_architecture(&pi05).as_deref(), Some("pi05"));

        std::fs::remove_file(smolvla).ok();
        std::fs::remove_file(pi05).ok();
    }

    #[test]
    fn malformed_or_non_vla_artifacts_fail_closed() {
        let malformed = temp_file("advisor-vla-malformed", "gguf");
        std::fs::write(&malformed, b"GGUF\x03").unwrap();
        assert_eq!(detect_vla_architecture(&malformed), None);

        let chat = temp_file("advisor-chat", "gguf");
        write_test_gguf(&chat, &[("general.architecture", "qwen3")]);
        assert_eq!(detect_vla_architecture(&chat), None);

        std::fs::remove_file(malformed).ok();
        std::fs::remove_file(chat).ok();
    }

    fn model_has_capability(payload: &Value, wanted: &str) -> bool {
        payload["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|capability| capability.as_str() == Some(wanted))
    }

    fn temp_file(name: &str, extension: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("omniinfer-{name}-{nanos}.{extension}"))
    }

    fn write_test_gguf(path: &Path, metadata: &[(&str, &str)]) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3_u32.to_le_bytes()).unwrap();
        file.write_all(&0_u64.to_le_bytes()).unwrap();
        file.write_all(&(metadata.len() as u64).to_le_bytes())
            .unwrap();
        for (key, value) in metadata {
            write_gguf_string(&mut file, key);
            file.write_all(&8_u32.to_le_bytes()).unwrap();
            write_gguf_string(&mut file, value);
        }
    }

    fn write_gguf_string(file: &mut std::fs::File, value: &str) {
        file.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        file.write_all(value.as_bytes()).unwrap();
    }

    fn write_test_safetensors(path: &Path, header: &[u8]) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
    }
}
