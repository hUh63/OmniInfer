use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalRuntimeRequest {
    pub backend: Value,
    pub model_path: String,
    pub mmproj_path: Option<String>,
    pub host: String,
    pub port: u16,
    pub ctx_size: Option<u32>,
    pub launch_args: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalRuntimePlan {
    pub command: Vec<String>,
    pub stop_command: Option<Vec<String>>,
    pub cwd: PathBuf,
    pub port: u16,
    pub ctx_size: Option<u32>,
    pub log_file_name: String,
    pub proxy_model_ref: Option<String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RuntimePlanError {
    #[error("backend payload is missing field: {0}")]
    MissingBackendField(&'static str),
    #[error("backend launcher not found: {0}")]
    MissingLauncher(String),
    #[error("launch arg {0:?} is managed by OmniInfer and must not be set in backend config")]
    ReservedLaunchArg(String),
    #[error("unsupported external runtime protocol for {backend}: {protocol}")]
    UnsupportedProtocol { backend: String, protocol: String },
    #[error("port must be in 1-65535")]
    InvalidPort,
    #[error("invalid WSL2 launcher manifest {path}: {message}")]
    InvalidWslLauncherManifest { path: String, message: String },
    #[error("WSL2 vLLM does not support this Windows model path: {0}")]
    UnsupportedWslModelPath(String),
}

#[derive(Debug, Deserialize)]
struct WslVllmLauncherManifest {
    schema_version: u32,
    backend: String,
    distribution: String,
    linux_launcher: String,
    linux_runner: String,
    linux_stopper: String,
    linux_pid_dir: String,
    automount_root: String,
}

pub fn build_external_runtime_plan(
    request: &ExternalRuntimeRequest,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    if request.port == 0 {
        return Err(RuntimePlanError::InvalidPort);
    }
    let backend_id = required_str(&request.backend, "id")?;
    let protocol =
        optional_str(&request.backend, "external_server_protocol").unwrap_or("llama.cpp-server");
    let launcher = optional_str(&request.backend, "launcher_path")
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| RuntimePlanError::MissingLauncher(backend_id.to_string()))?;
    let launcher_path = PathBuf::from(launcher);
    let mut server_args = request
        .launch_args
        .clone()
        .unwrap_or_else(|| string_array(&request.backend, "default_args"));
    validate_launch_args(&server_args)?;
    let ctx_flags = ctx_size_flags(protocol);
    if let Some(ctx_size) = request.ctx_size {
        server_args = with_server_arg(server_args, &ctx_flags, ctx_size.to_string());
    }
    let effective_ctx_size =
        extract_server_arg_value(&server_args, &ctx_flags).and_then(|value| value.parse().ok());
    let log_file_name = optional_str(&request.backend, "log_file_name")
        .unwrap_or("runtime.log")
        .to_string();

    match protocol {
        "llama.cpp-server" => build_llama_cpp_plan(
            backend_id,
            &launcher_path,
            request,
            server_args,
            effective_ctx_size,
            log_file_name,
        ),
        "vllm-openai-server" => build_vllm_plan(
            &launcher_path,
            request,
            server_args,
            effective_ctx_size,
            log_file_name,
        ),
        "vllm-wsl2-openai-server" => build_wsl_vllm_plan(
            backend_id,
            &launcher_path,
            request,
            server_args,
            effective_ctx_size,
            log_file_name,
        ),
        other => Err(RuntimePlanError::UnsupportedProtocol {
            backend: backend_id.to_string(),
            protocol: other.to_string(),
        }),
    }
}

fn build_llama_cpp_plan(
    backend_id: &str,
    launcher_path: &Path,
    request: &ExternalRuntimeRequest,
    server_args: Vec<String>,
    effective_ctx_size: Option<u32>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    let mut command = vec![
        launcher_path.display().to_string(),
        "-m".to_string(),
        request.model_path.clone(),
        "--host".to_string(),
        request.host.clone(),
        "--port".to_string(),
        request.port.to_string(),
    ];
    if backend_id.starts_with("ik_llama.cpp") {
        command.extend(["--webui".to_string(), "none".to_string()]);
    } else {
        command.push("--no-webui".to_string());
    }
    let log_dir = runtime_dir(&request.backend).join("logs");
    command.extend([
        "--slot-save-path".to_string(),
        log_dir.display().to_string(),
    ]);
    command.extend(server_args);
    if let Some(mmproj) = request
        .mmproj_path
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        command.extend(["--mmproj".to_string(), mmproj.to_string()]);
    }
    Ok(ExternalRuntimePlan {
        command,
        stop_command: None,
        cwd: launcher_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: effective_ctx_size,
        log_file_name,
        proxy_model_ref: None,
    })
}

fn build_vllm_plan(
    launcher_path: &Path,
    request: &ExternalRuntimeRequest,
    mut server_args: Vec<String>,
    effective_ctx_size: Option<u32>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    if extract_server_arg_value(&server_args, &["--served-model-name"]).is_none() {
        server_args.splice(
            0..0,
            ["--served-model-name".to_string(), "local".to_string()],
        );
    }
    let proxy_model_ref = extract_server_arg_value(&server_args, &["--served-model-name"]);
    let mut command = vec![
        launcher_path.display().to_string(),
        "serve".to_string(),
        request.model_path.clone(),
        "--host".to_string(),
        request.host.clone(),
        "--port".to_string(),
        request.port.to_string(),
    ];
    command.extend(server_args);
    Ok(ExternalRuntimePlan {
        command,
        stop_command: None,
        cwd: launcher_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: effective_ctx_size,
        log_file_name,
        proxy_model_ref,
    })
}

fn build_wsl_vllm_plan(
    backend_id: &str,
    manifest_path: &Path,
    request: &ExternalRuntimeRequest,
    mut server_args: Vec<String>,
    effective_ctx_size: Option<u32>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    let raw = std::fs::read_to_string(manifest_path).map_err(|error| {
        RuntimePlanError::InvalidWslLauncherManifest {
            path: manifest_path.display().to_string(),
            message: error.to_string(),
        }
    })?;
    let manifest: WslVllmLauncherManifest = serde_json::from_str(&raw).map_err(|error| {
        RuntimePlanError::InvalidWslLauncherManifest {
            path: manifest_path.display().to_string(),
            message: error.to_string(),
        }
    })?;
    validate_wsl_manifest(backend_id, manifest_path, &manifest)?;
    if extract_server_arg_value(&server_args, &["--served-model-name"]).is_none() {
        server_args.splice(
            0..0,
            ["--served-model-name".to_string(), "local".to_string()],
        );
    }
    let proxy_model_ref = extract_server_arg_value(&server_args, &["--served-model-name"]);
    let model = translate_wsl_model_ref(&request.model_path, &manifest.automount_root)?;
    let pid_file = format!(
        "{}/{}.pid",
        manifest.linux_pid_dir.trim_end_matches('/'),
        request.port
    );
    let wsl = std::env::var("OMNIINFER_WSL_EXE").unwrap_or_else(|_| "wsl.exe".to_string());
    let mut command = vec![
        wsl.clone(),
        "--distribution".to_string(),
        manifest.distribution.clone(),
        "--exec".to_string(),
        manifest.linux_runner.clone(),
        pid_file.clone(),
        manifest.linux_launcher.clone(),
        "serve".to_string(),
        model,
        "--host".to_string(),
        request.host.clone(),
        "--port".to_string(),
        request.port.to_string(),
    ];
    command.extend(server_args);
    let stop_command = vec![
        wsl,
        "--distribution".to_string(),
        manifest.distribution,
        "--exec".to_string(),
        manifest.linux_stopper,
        pid_file,
    ];
    Ok(ExternalRuntimePlan {
        command,
        stop_command: Some(stop_command),
        cwd: manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: effective_ctx_size,
        log_file_name,
        proxy_model_ref,
    })
}

fn validate_wsl_manifest(
    backend_id: &str,
    path: &Path,
    manifest: &WslVllmLauncherManifest,
) -> Result<(), RuntimePlanError> {
    let valid = manifest.schema_version == 1
        && manifest.backend == backend_id
        && !manifest.distribution.trim().is_empty()
        && !manifest.distribution.chars().any(char::is_control)
        && [
            manifest.linux_launcher.as_str(),
            manifest.linux_runner.as_str(),
            manifest.linux_stopper.as_str(),
            manifest.linux_pid_dir.as_str(),
            manifest.automount_root.as_str(),
        ]
        .into_iter()
        .all(valid_absolute_linux_path);
    if valid {
        Ok(())
    } else {
        Err(RuntimePlanError::InvalidWslLauncherManifest {
            path: path.display().to_string(),
            message: "unsupported schema or incomplete absolute Linux paths".to_string(),
        })
    }
}

fn valid_absolute_linux_path(value: &str) -> bool {
    value.starts_with('/')
        && !value.chars().any(char::is_control)
        && !value.split('/').any(|component| component == "..")
}

fn translate_wsl_model_ref(model: &str, automount_root: &str) -> Result<String, RuntimePlanError> {
    let bytes = model.as_bytes();
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/')
    {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        let suffix = model[3..].replace('\\', "/");
        return Ok(format!(
            "{}/{drive}/{}",
            automount_root.trim_end_matches('/'),
            suffix.trim_start_matches('/')
        ));
    }
    if model.starts_with(r"\\") {
        return Err(RuntimePlanError::UnsupportedWslModelPath(model.to_string()));
    }
    Ok(model.to_string())
}

fn validate_launch_args(args: &[String]) -> Result<(), RuntimePlanError> {
    for token in args {
        let flag = token.split_once('=').map(|(flag, _)| flag).unwrap_or(token);
        if matches!(
            flag,
            "-m" | "--model" | "-mm" | "--mmproj" | "--host" | "--port" | "--no-webui"
        ) {
            return Err(RuntimePlanError::ReservedLaunchArg(flag.to_string()));
        }
    }
    Ok(())
}

fn with_server_arg(mut args: Vec<String>, flags: &[&str], value: String) -> Vec<String> {
    let mut updated = Vec::with_capacity(args.len() + 2);
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        if flags.contains(&token.as_str()) {
            index += if index + 1 < args.len() { 2 } else { 1 };
            continue;
        }
        updated.push(std::mem::take(&mut args[index]));
        index += 1;
    }
    updated.extend([flags[0].to_string(), value]);
    updated
}

fn extract_server_arg_value(args: &[String], flags: &[&str]) -> Option<String> {
    let mut value = None;
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        if flags.contains(&token.as_str()) {
            if let Some(next) = args.get(index + 1) {
                value = Some(next.clone());
            }
            index += 2;
            continue;
        }
        index += 1;
    }
    value
}

fn ctx_size_flags(protocol: &str) -> [&'static str; 2] {
    if matches!(protocol, "vllm-openai-server" | "vllm-wsl2-openai-server") {
        ["--max-model-len", ""]
    } else {
        ["-c", "--ctx-size"]
    }
}

fn required_str<'a>(value: &'a Value, key: &'static str) -> Result<&'a str, RuntimePlanError> {
    optional_str(value, key).ok_or(RuntimePlanError::MissingBackendField(key))
}

fn optional_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
}

fn string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn runtime_dir(backend: &Value) -> PathBuf {
    optional_str(backend, "runtime_dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn builds_llama_cpp_cuda_command() {
        let backend = json!({
            "id": "llama.cpp-linux-cuda",
            "launcher_path": "/runtime/llama.cpp-linux-cuda/bin/llama-server",
            "runtime_dir": "/runtime/llama.cpp-linux-cuda",
            "default_args": ["-ngl", "999"],
            "external_server_protocol": "llama.cpp-server",
            "log_file_name": "runtime.log"
        });
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend,
            model_path: "/models/qwen.gguf".to_string(),
            mmproj_path: Some("/models/mmproj.gguf".to_string()),
            host: "127.0.0.1".to_string(),
            port: 12345,
            ctx_size: Some(8192),
            launch_args: None,
        })
        .unwrap();
        let log_dir = PathBuf::from("/runtime/llama.cpp-linux-cuda")
            .join("logs")
            .display()
            .to_string();
        assert_eq!(plan.ctx_size, Some(8192));
        assert_eq!(plan.cwd, PathBuf::from("/runtime/llama.cpp-linux-cuda/bin"));
        assert_eq!(
            plan.command,
            vec![
                "/runtime/llama.cpp-linux-cuda/bin/llama-server".to_string(),
                "-m".to_string(),
                "/models/qwen.gguf".to_string(),
                "--host".to_string(),
                "127.0.0.1".to_string(),
                "--port".to_string(),
                "12345".to_string(),
                "--no-webui".to_string(),
                "--slot-save-path".to_string(),
                log_dir,
                "-ngl".to_string(),
                "999".to_string(),
                "-c".to_string(),
                "8192".to_string(),
                "--mmproj".to_string(),
                "/models/mmproj.gguf".to_string()
            ]
        );
    }

    #[test]
    fn ik_uses_webui_none() {
        let backend = json!({
            "id": "ik_llama.cpp-linux-cuda",
            "launcher_path": "/runtime/ik/bin/llama-server",
            "runtime_dir": "/runtime/ik",
            "default_args": ["--jinja", "-ngl", "999"],
            "external_server_protocol": "llama.cpp-server"
        });
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend,
            model_path: "/models/qwen.gguf".to_string(),
            mmproj_path: None,
            host: "127.0.0.1".to_string(),
            port: 12345,
            ctx_size: None,
            launch_args: None,
        })
        .unwrap();
        assert!(
            plan.command
                .windows(2)
                .any(|items| items == ["--webui", "none"])
        );
        assert!(!plan.command.iter().any(|item| item == "--no-webui"));
    }

    #[test]
    fn vllm_uses_openai_server_shape() {
        let backend = json!({
            "id": "vllm-linux-cuda",
            "launcher_path": "/runtime/vllm/bin/vllm",
            "runtime_dir": "/runtime/vllm",
            "default_args": ["--max-model-len", "4096"],
            "external_server_protocol": "vllm-openai-server",
            "log_file_name": "vllm-server.log"
        });
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend,
            model_path: "Qwen/Qwen3.5-4B".to_string(),
            mmproj_path: None,
            host: "127.0.0.1".to_string(),
            port: 23456,
            ctx_size: None,
            launch_args: None,
        })
        .unwrap();
        assert_eq!(plan.ctx_size, Some(4096));
        assert_eq!(plan.proxy_model_ref.as_deref(), Some("local"));
        assert_eq!(
            plan.command,
            vec![
                "/runtime/vllm/bin/vllm",
                "serve",
                "Qwen/Qwen3.5-4B",
                "--host",
                "127.0.0.1",
                "--port",
                "23456",
                "--served-model-name",
                "local",
                "--max-model-len",
                "4096"
            ]
        );
    }

    #[test]
    fn wsl_vllm_uses_manifest_and_translates_windows_model_path() {
        let root = std::env::temp_dir().join(format!("omniinfer-wsl-plan-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let manifest = root.join("vllm-wsl2.json");
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({
                "schema_version": 1,
                "backend": "vllm-wsl2-cuda",
                "distribution": "Ubuntu-24.04",
                "linux_launcher": "/home/test/runtime/current/venv/bin/vllm",
                "linux_runner": "/home/test/runtime/current/bin/omniinfer-vllm-run",
                "linux_stopper": "/home/test/runtime/current/bin/omniinfer-vllm-stop",
                "linux_pid_dir": "/home/test/runtime/current/run",
                "automount_root": "/mnt",
            }))
            .unwrap(),
        )
        .unwrap();
        let backend = json!({
            "id": "vllm-wsl2-cuda",
            "launcher_path": manifest,
            "runtime_dir": root,
            "default_args": [],
            "external_server_protocol": "vllm-wsl2-openai-server",
            "log_file_name": "vllm-wsl2-server.log"
        });
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend,
            model_path: r"D:\models\Qwen 4B".to_string(),
            mmproj_path: None,
            host: "127.0.0.1".to_string(),
            port: 24567,
            ctx_size: Some(8192),
            launch_args: None,
        })
        .unwrap();
        assert_eq!(plan.proxy_model_ref.as_deref(), Some("local"));
        assert!(
            plan.command
                .iter()
                .any(|arg| arg == "/mnt/d/models/Qwen 4B")
        );
        assert!(
            plan.command
                .windows(2)
                .any(|args| args == ["--max-model-len", "8192"])
        );
        assert_eq!(
            plan.stop_command.as_ref().unwrap().last().unwrap(),
            "/home/test/runtime/current/run/24567.pid"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn wsl_vllm_rejects_unc_model_path() {
        let error = translate_wsl_model_ref(r"\\server\share\model", "/mnt").unwrap_err();
        assert_eq!(
            error,
            RuntimePlanError::UnsupportedWslModelPath(r"\\server\share\model".to_string())
        );
    }

    #[test]
    fn wsl_vllm_keeps_huggingface_references_and_rejects_malformed_manifest() {
        assert_eq!(
            translate_wsl_model_ref("Qwen/Qwen2.5-7B-Instruct", "/mnt").unwrap(),
            "Qwen/Qwen2.5-7B-Instruct"
        );
        let manifest = WslVllmLauncherManifest {
            schema_version: 1,
            backend: "vllm-wsl2-cuda".to_string(),
            distribution: "Ubuntu-24.04".to_string(),
            linux_launcher: "/home/test/runtime/../other/vllm".to_string(),
            linux_runner: "/home/test/runtime/run".to_string(),
            linux_stopper: "/home/test/runtime/stop".to_string(),
            linux_pid_dir: "/home/test/runtime/pids".to_string(),
            automount_root: "/mnt".to_string(),
        };
        let error = validate_wsl_manifest("vllm-wsl2-cuda", Path::new("vllm-wsl2.json"), &manifest)
            .unwrap_err();
        assert!(matches!(
            error,
            RuntimePlanError::InvalidWslLauncherManifest { .. }
        ));
    }

    #[test]
    fn ctx_size_replaces_existing_flag() {
        let backend = json!({
            "id": "llama.cpp-linux-cuda",
            "launcher_path": "/runtime/bin/llama-server",
            "runtime_dir": "/runtime",
            "default_args": ["-ngl", "999", "--ctx-size", "2048"],
            "external_server_protocol": "llama.cpp-server"
        });
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend,
            model_path: "/models/qwen.gguf".to_string(),
            mmproj_path: None,
            host: "127.0.0.1".to_string(),
            port: 12345,
            ctx_size: Some(8192),
            launch_args: None,
        })
        .unwrap();
        assert_eq!(plan.ctx_size, Some(8192));
        assert!(plan.command.windows(2).any(|items| items == ["-c", "8192"]));
        assert!(
            !plan
                .command
                .windows(2)
                .any(|items| items == ["--ctx-size", "2048"])
        );
    }

    #[test]
    fn rejects_reserved_managed_args() {
        let backend = json!({
            "id": "llama.cpp-linux-cuda",
            "launcher_path": "/runtime/bin/llama-server",
            "runtime_dir": "/runtime",
            "default_args": ["--host", "0.0.0.0"],
            "external_server_protocol": "llama.cpp-server"
        });
        let error = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend,
            model_path: "/models/qwen.gguf".to_string(),
            mmproj_path: None,
            host: "127.0.0.1".to_string(),
            port: 12345,
            ctx_size: None,
            launch_args: None,
        })
        .unwrap_err();
        assert_eq!(
            error,
            RuntimePlanError::ReservedLaunchArg("--host".to_string())
        );
    }
}
