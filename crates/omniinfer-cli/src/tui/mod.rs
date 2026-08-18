use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::Result;
use omniinfer_core::{
    chat_stream, config, http_client, local_state, model_load, paths, serve_state,
};
use serde_json::Value;

mod models;
mod render;

use crate::{
    BackendScope, ServeArgs, advisor, backend_installer, get_local_json_for_config, json_bool,
    json_str, json_u64, load_model_with_request_for_config, post_local_json_for_config,
    print_chat_performance, rust_backend_payload, select_backend_for_config, serve_orchestrated,
    stop_serve, wait_for_gateway_ready,
};

use models::{
    advisor_model_summary, advisor_recommendation_map, discover_local_models, model_context_label,
    model_provider_label, model_quant_label, model_size_label, prompt_model_path, same_path,
};
use render::{
    ModelMenuContext, ModelMenuItem, NoticeKind, clear_screen, is_interactive, notice,
    print_chat_header, print_header, print_help, print_kv, print_section, prompt_default,
    select_menu, select_model_menu,
};

#[derive(Debug, Clone)]
struct MenuItem {
    label: String,
    details: Vec<String>,
    selected: bool,
}

#[derive(Debug)]
struct ChatSession {
    backend: String,
    reasoning_visible: bool,
    messages: Vec<Value>,
    last_usage: Option<Value>,
}

pub fn run() -> Result<()> {
    if !is_interactive() {
        anyhow::bail!("OmniInfer TUI requires an interactive terminal.");
    }
    clear_screen();
    print_header("OmniInfer", "Local inference console");
    let config = config::load_app_config().unwrap_or_default();
    let _gateway = TuiGatewayGuard::ensure(&config)?;
    let state = local_state::load_state().unwrap_or_default();
    let backend = match state.selected_model.clone() {
        Some(model) if Path::new(&model.model).exists() => {
            match load_remembered_model(&config, &model) {
                Ok(backend) => backend,
                Err(error) => {
                    notice(
                        &format!("Could not load previous model: {error}"),
                        NoticeKind::Warning,
                    );
                    setup_model_flow(&config)?
                }
            }
        }
        _ => setup_model_flow(&config)?,
    };
    chat_loop(&config, backend)?;
    Ok(())
}

struct TuiGatewayGuard {
    port: u16,
    owned: bool,
    child: Option<Child>,
    stopped: Arc<AtomicBool>,
}

impl TuiGatewayGuard {
    fn ensure(config: &config::AppConfig) -> Result<Self> {
        if get_running_state(config).is_some() {
            return Ok(Self {
                port: config.port,
                owned: false,
                child: None,
                stopped: Arc::new(AtomicBool::new(false)),
            });
        }
        print_section("Service", "Starting local OmniInfer gateway");
        print_kv("Port", &config.port.to_string());
        let mut command = ProcessCommand::new(std::env::current_exe()?);
        paths::propagate_cli_roots(&mut command);
        command
            .arg("gateway")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(config.port.to_string())
            .current_dir(paths::repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let child = command.spawn()?;
        let guard = Self {
            port: config.port,
            owned: true,
            child: Some(child),
            stopped: Arc::new(AtomicBool::new(false)),
        };
        wait_for_gateway_ready(config)?;
        guard.install_ctrl_c_handler();
        notice("Local gateway ready", NoticeKind::Success);
        println!();
        Ok(guard)
    }

    fn install_ctrl_c_handler(&self) {
        if !self.owned {
            return;
        }
        let port = self.port;
        let stopped = Arc::clone(&self.stopped);
        std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            runtime.block_on(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    stop_tui_owned_gateway(port, &stopped);
                    std::process::exit(130);
                }
            });
        });
    }
}

impl Drop for TuiGatewayGuard {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        stop_tui_owned_gateway(self.port, &self.stopped);
        if let Some(child) = self.child.as_mut() {
            let _ = child.try_wait();
        }
    }
}

fn stop_tui_owned_gateway(port: u16, stopped: &AtomicBool) {
    if stopped.swap(true, Ordering::SeqCst) {
        return;
    }
    let _ = stop_serve(port);
}

pub fn run_server(args: &ServeArgs) -> Result<()> {
    if !is_interactive() {
        return serve_orchestrated(args);
    }
    clear_screen();
    print_header("OmniInfer Server", "Interactive gateway launcher");
    let config = config::load_app_config().unwrap_or_default();
    let backend =
        choose_backend(&config)?.ok_or_else(|| anyhow::anyhow!("No backend selected."))?;
    let model =
        choose_model(&config, true)?.ok_or_else(|| anyhow::anyhow!("No model selected."))?;
    let mut args = args.clone();
    args.backend = Some(backend);
    args.model = Some(model.display().to_string());
    serve_orchestrated(&args)
}

mod model_flow;

use model_flow::*;
mod chat_session;

use chat_session::chat_loop;
#[allow(dead_code)]
fn _loaded_services() -> Vec<serve_state::ServePidInfo> {
    serve_state::list_serve_pid_infos().unwrap_or_default()
}

#[cfg(test)]
mod backend_model_tests {
    use super::*;

    #[test]
    fn vla_backend_accepts_only_supported_checkpoint_files() {
        let backend = SelectedBackendInfo {
            family: "vla.cpp".to_string(),
            model_artifact: "vla-artifact".to_string(),
        };
        assert!(model_supported_by_backend(
            Path::new("smolvla.gguf"),
            Some(&backend)
        ));
        assert!(model_supported_by_backend(
            Path::new("smolvla.safetensors"),
            Some(&backend)
        ));
        assert!(!model_supported_by_backend(
            Path::new("weights.bin"),
            Some(&backend)
        ));
        assert!(!model_supported_by_backend(
            Path::new("config.json"),
            Some(&backend)
        ));
    }

    #[test]
    fn chat_backends_do_not_claim_vla_safetensors() {
        let backend = SelectedBackendInfo {
            family: "llama.cpp".to_string(),
            model_artifact: "file".to_string(),
        };
        assert!(model_supported_by_backend(
            Path::new("chat.gguf"),
            Some(&backend)
        ));
        assert!(!model_supported_by_backend(
            Path::new("weights.safetensors"),
            Some(&backend)
        ));
    }

    #[test]
    fn remembered_model_reuse_requires_matching_request_defaults() {
        let model = local_state::SelectedModel {
            model: "/models/model.gguf".to_string(),
            mmproj: None,
            ctx_size: Some(4096),
            request_defaults: serde_json::from_value(serde_json::json!({"max_tokens": 64}))
                .unwrap(),
        };
        assert!(state_matches_remembered_model(
            &serde_json::json!({
                "model_path": "/models/model.gguf",
                "request_defaults": {"max_tokens": 64}
            }),
            &model,
        ));
        assert!(!state_matches_remembered_model(
            &serde_json::json!({
                "model_path": "/models/model.gguf",
                "request_defaults": {"max_tokens": 128}
            }),
            &model,
        ));
    }
}
