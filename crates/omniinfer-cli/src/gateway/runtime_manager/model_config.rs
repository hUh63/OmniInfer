use super::*;

pub(super) fn model_log_file_name(base: &str, model_key: &str) -> String {
    let sanitized = model_key
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    match base.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && !ext.is_empty() => {
            format!("{stem}-{sanitized}.{ext}")
        }
        _ => format!("{base}-{sanitized}.log"),
    }
}

pub(super) fn json_required_str<'a>(payload: &'a Value, key: &'static str) -> Result<&'a str> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("field '{key}' is required"))
}

pub(super) fn request_defaults_from_payload(payload: &Value) -> Result<Map<String, Value>> {
    match payload.get("request_defaults") {
        Some(Value::Object(defaults)) => Ok(defaults.clone()),
        Some(_) => anyhow::bail!("request_defaults must be an object"),
        None => Ok(Map::new()),
    }
}

pub(super) fn resolve_model_for_backend(
    model: &str,
    backend: &backend_registry::BackendSpec,
) -> Result<omniinfer_core::model_artifacts::ResolvedModelArtifacts> {
    if backend.model_artifact == "reference" {
        return Ok(omniinfer_core::model_artifacts::ResolvedModelArtifacts {
            model_path: model.to_string(),
            mmproj_path: None,
        });
    }
    let path = resolve_path_for_backend(model, backend, "model")?;
    if backend.model_artifact == "vla-artifact" {
        let path = PathBuf::from(&path);
        if path.is_dir() {
            anyhow::bail!(
                "vla.cpp model must be a checkpoint file, not a directory: {}",
                path.display()
            );
        }
        if !is_vla_checkpoint_path(&path) {
            anyhow::bail!(
                "vla.cpp model must be a .gguf or .safetensors checkpoint: {}",
                path.display()
            );
        }
    }
    if backend.model_artifact == "file" && PathBuf::from(&path).is_dir() {
        return Ok(discover_llama_cpp_model_artifacts(&PathBuf::from(path))?);
    }
    Ok(omniinfer_core::model_artifacts::ResolvedModelArtifacts {
        model_path: path,
        mmproj_path: None,
    })
}

pub(super) fn is_vla_checkpoint_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("gguf") || extension.eq_ignore_ascii_case("safetensors")
        })
}

pub(super) fn resolve_path_for_backend(
    text: &str,
    backend: &backend_registry::BackendSpec,
    label: &str,
) -> Result<String> {
    let mut path = expand_home(PathBuf::from(text.trim()));
    if !path.is_absolute() {
        let Some(models_dir) = backend.models_dir.as_deref() else {
            anyhow::bail!("relative {label} path requires a configured models_dir");
        };
        path = PathBuf::from(models_dir).join(path);
    }
    if label == "model" && backend.model_artifact == "directory" {
        if !path.is_dir() {
            anyhow::bail!("model directory not found: {}", path.display());
        }
    } else if !path.exists() {
        anyhow::bail!("{label} not found: {}", path.display());
    }
    Ok(path.display().to_string())
}

pub(super) fn expand_home(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path
}

pub(super) fn launch_args_have_ctx_size(family: &str, args: &[String]) -> bool {
    args.iter().any(|arg| {
        let flag = arg.split_once('=').map(|(flag, _)| flag).unwrap_or(arg);
        match family {
            "vllm" => flag == "--max-model-len",
            "llama.cpp" | "turboquant" => matches!(flag, "-c" | "--ctx-size"),
            _ => matches!(flag, "-c" | "--ctx-size" | "--max-model-len"),
        }
    })
}

pub(super) fn merged_launch_args(
    backend_id: &str,
    family: &str,
    defaults: &[String],
    requested: Option<&[String]>,
) -> Vec<String> {
    let Some(requested) = requested else {
        return defaults.to_vec();
    };
    if family != "llama.cpp" || !backend_id.starts_with("llama.cpp-") {
        return requested.to_vec();
    }
    defaults.iter().chain(requested).cloned().collect()
}
