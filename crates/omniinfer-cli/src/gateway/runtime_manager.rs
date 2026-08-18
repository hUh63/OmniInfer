use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use omniinfer_core::backend_args::parse_backend_load_extra_args;
use omniinfer_core::backend_registry::{self, BackendRegistry, BackendScope};
use omniinfer_core::local_state;
use omniinfer_core::model_artifacts::{discover_llama_cpp_model_artifacts, maybe_auto_mmproj};
use omniinfer_core::model_load::DEFAULT_LOAD_CONTEXT_SIZE;
use omniinfer_core::resource_ledger::{
    AllocationId, BudgetComponent, MemoryDomain, ReservationId, ResourceBudget, ResourceCapacity,
    ResourceLedger,
};
use omniinfer_core::runtime_plan::{
    ExternalRuntimeRequest, ExternalServerProtocol, build_external_runtime_plan,
};
use omniinfer_core::runtime_process::{RuntimeProcess, RuntimeProcessError, RuntimeProcessOptions};
use serde_json::{Value, json};

use super::gpu_status::runtime_env_for_backend;

const WSL_ROCM_COLD_START_RETRY_MINIMUM_BUDGET: Duration = Duration::from_secs(360);
const WSL_ROCM_COLD_START_INITIAL_ATTEMPT: Duration = Duration::from_secs(120);
const WSL_ROCM_COLD_START_RETRY_COOLDOWN: Duration = Duration::from_secs(90);

pub(super) struct RustRuntimeManager {
    selected_backend: Option<String>,
    loaded: BTreeMap<String, LoadedRustRuntime>,
    default_model_key: Option<String>,
    resource_ledger: Option<ResourceLedger>,
    next_capacity_snapshot: u64,
    next_generation: u64,
}

impl Default for RustRuntimeManager {
    fn default() -> Self {
        Self {
            selected_backend: None,
            loaded: BTreeMap::new(),
            default_model_key: None,
            resource_ledger: None,
            next_capacity_snapshot: 1,
            next_generation: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuntimeProxyTarget {
    pub(super) base_url: Option<String>,
    pub(super) client_endpoint: String,
    pub(super) protocol: ExternalServerProtocol,
    pub(super) backend_id: String,
    pub(super) model: Option<String>,
    pub(super) generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeRouteState {
    Ready,
    Draining,
    Failed,
}

impl RuntimeRouteState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Failed => "failed",
        }
    }
}

struct LoadedRustRuntime {
    model_key: String,
    owner_admin_id: Option<String>,
    backend_id: String,
    model: String,
    public_model_id: Option<String>,
    mmproj: Option<String>,
    ctx_size: Option<u32>,
    launch_args: Vec<String>,
    cuda_visible_devices: Option<String>,
    cuda_warning: Option<String>,
    external_server_protocol: ExternalServerProtocol,
    client_endpoint: String,
    process: RuntimeProcess,
    proxy_model_ref: Option<String>,
    generation: u64,
    route_state: RuntimeRouteState,
    allocation_id: AllocationId,
    resource_budget: ResourceBudget,
}

#[derive(Debug, Clone)]
pub(super) struct LoadedRuntimeSummary {
    pub(super) id: String,
    pub(super) owner_admin_id: Option<String>,
    pub(super) backend_pid: u32,
}

pub(super) enum LoadModelOutcome {
    Success(Value),
    ReloadRequired(Value),
}

impl RustRuntimeManager {
    pub(super) fn select_backend(&mut self, backend_id: &str) -> Result<Value> {
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(backend_id)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {backend_id}"))?;
        if self.selected_backend.as_deref() != Some(backend_id) {
            self.stop_runtime()?;
        }
        self.selected_backend = Some(backend_id.to_string());
        local_state::save_selected_backend(backend_id)?;
        Ok(json!({
            "ok": true,
            "selected_backend": backend_id,
            "binary_exists": backend.binary_exists(),
            "models_dir": backend.models_dir,
        }))
    }

    pub(super) fn stop_runtime(&mut self) -> Result<Value> {
        let keys = self.loaded.keys().cloned().collect::<Vec<_>>();
        let mut failures = Vec::new();
        for key in keys {
            let stop_result = {
                let loaded = self
                    .loaded
                    .get_mut(&key)
                    .expect("runtime key came from the loaded map");
                loaded.route_state = RuntimeRouteState::Draining;
                loaded.process.stop(Duration::from_secs(8))
            };
            match stop_result {
                Ok(()) => self.remove_runtime_and_release(&key),
                Err(error) => {
                    if let Some(loaded) = self.loaded.get_mut(&key) {
                        loaded.route_state = RuntimeRouteState::Failed;
                    }
                    failures.push(format!("{key}: {error}"));
                }
            }
        }
        self.select_fallback_default();
        if !failures.is_empty() {
            anyhow::bail!("failed to stop runtimes: {}", failures.join("; "));
        }
        let selected_model_preserved = local_state::load_state()
            .ok()
            .is_some_and(|state| state.selected_model.is_some());
        Ok(json!({
            "ok": true,
            "stopped": true,
            "selected_backend": self.selected_backend,
            "selected_model_preserved": selected_model_preserved,
            "restore_status": if selected_model_preserved { "pending" } else { "not_configured" },
        }))
    }

    pub(super) fn has_loaded_runtime(&mut self) -> bool {
        self.reap_exited_runtimes();
        self.loaded
            .values()
            .any(|loaded| loaded.route_state == RuntimeRouteState::Ready)
    }

    pub(super) fn load_model(
        &mut self,
        payload: Value,
        backend_host: String,
        startup_timeout: Duration,
        owner_admin_id: Option<String>,
        startup_cancelled: &AtomicBool,
    ) -> Result<LoadModelOutcome> {
        if startup_cancelled.load(Ordering::SeqCst) {
            anyhow::bail!("gateway is shutting down")
        }
        self.reap_exited_runtimes();
        let model = json_required_str(&payload, "model")?.to_string();
        let public_model_id = payload
            .get("public_model_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string);
        let requested_model_key = public_model_id.clone().unwrap_or_else(|| model.clone());
        let requested_backend = self.resolve_requested_backend(&payload)?;
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(&requested_backend)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {requested_backend}"))?;
        if backend.runtime_mode != "external_server" {
            anyhow::bail!(
                "{} is an embedded backend. Python control-plane fallback has been removed; use an external-server backend or a backend adapter service.",
                backend.id
            );
        }
        if !backend.binary_exists() {
            anyhow::bail!(
                "backend launcher not found: {}",
                backend.launcher_path.as_deref().unwrap_or("(unset)")
            );
        }
        let resolved_model = resolve_model_for_backend(&model, backend)?;
        let explicit_mmproj = payload
            .get("mmproj")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(|value| resolve_path_for_backend(value, backend, "mmproj file"))
            .transpose()?;
        let mmproj_path = explicit_mmproj.or(resolved_model.mmproj_path).or_else(|| {
            maybe_auto_mmproj(backend.models_dir.as_deref(), &resolved_model.model_path)
        });
        if mmproj_path.is_some() && !backend.supports_mmproj {
            anyhow::bail!("{} does not support mmproj inputs", backend.id);
        }
        let requested_ctx_size = payload
            .get("ctx_size")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());
        let launch_args = payload
            .get("launch_args")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            });
        let effective_launch_args = merged_launch_args(
            &backend.id,
            &backend.family,
            &backend.default_args,
            launch_args.as_deref(),
        );
        let launch_args_have_ctx =
            launch_args_have_ctx_size(&backend.family, &effective_launch_args);
        let launch_args_ctx_size =
            parse_backend_load_extra_args(&backend.id, &backend.family, &effective_launch_args)
                .ok()
                .and_then(|parsed| parsed.ctx_size);
        let ctx_size = requested_ctx_size.or(launch_args_ctx_size).or_else(|| {
            (backend.supports_ctx_size && !launch_args_have_ctx)
                .then_some(DEFAULT_LOAD_CONTEXT_SIZE)
        });
        if let Some(loaded_key) = self.matching_loaded_model_key(
            &requested_model_key,
            &resolved_model.model_path,
            public_model_id.as_deref(),
        ) {
            let loaded = self
                .loaded
                .get(&loaded_key)
                .expect("matched runtime should remain registered");
            if same_load_configuration(
                loaded,
                &backend.id,
                &resolved_model.model_path,
                mmproj_path.as_deref(),
                ctx_size,
                &effective_launch_args,
            ) {
                let loaded_key = self.promote_loaded_model_key(
                    &loaded_key,
                    &requested_model_key,
                    public_model_id.as_deref(),
                );
                let loaded = self
                    .loaded
                    .get(&loaded_key)
                    .expect("promoted runtime should remain registered");
                let response = model_load_response(loaded, true);
                self.default_model_key = Some(loaded_key);
                local_state::save_selected_backend(&backend.id)?;
                local_state::save_selected_model(
                    &resolved_model.model_path,
                    mmproj_path.as_deref(),
                    ctx_size,
                )?;
                return Ok(LoadModelOutcome::Success(response));
            }
            let requested = RequestedRuntimeConfig {
                backend_id: &backend.id,
                model_key: &requested_model_key,
                model_path: &resolved_model.model_path,
                public_model_id: public_model_id.as_deref(),
                mmproj: mmproj_path.as_deref(),
                ctx_size,
                launch_args: &effective_launch_args,
            };
            return Ok(LoadModelOutcome::ReloadRequired(reload_required_response(
                loaded, &requested,
            )));
        }
        let port = payload
            .get("backend_port")
            .and_then(Value::as_u64)
            .filter(|value| (1..=u64::from(u16::MAX)).contains(value))
            .and_then(|value| u16::try_from(value).ok())
            .map(Ok)
            .unwrap_or_else(|| pick_runtime_port(&backend_host))?;
        let backend_payload = serde_json::to_value(backend)?;
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend: backend_payload,
            model_path: resolved_model.model_path.clone(),
            mmproj_path: mmproj_path.clone(),
            host: backend_host.clone(),
            port,
            ctx_size,
            launch_args: Some(effective_launch_args.clone()),
        })?;
        let log_path = PathBuf::from(&backend.runtime_dir)
            .join("logs")
            .join(model_log_file_name(
                &plan.log_file_name,
                &requested_model_key,
            ));
        let (runtime_env, cuda_selection) =
            runtime_env_for_backend(backend, &effective_launch_args);
        let budget_cuda_devices = if backend.capabilities.iter().any(|value| value == "cuda") {
            match cuda_selection.as_ref() {
                Some(selection) => Some(selection.visible_devices.clone()),
                None => Some(detect_cuda_device_ids()?.join(",")),
            }
        } else {
            None
        };
        let resource_budget = build_runtime_resource_budget(
            &payload,
            backend,
            &resolved_model.model_path,
            mmproj_path.as_deref(),
            plan.ctx_size.unwrap_or(DEFAULT_LOAD_CONTEXT_SIZE),
            budget_cuda_devices.as_deref(),
            cuda_selection.is_none() && budget_cuda_devices.is_some(),
        )?;
        let reservation_id = self.reserve_runtime_resources(
            &requested_model_key,
            &resource_budget,
            budget_cuda_devices.as_deref(),
        )?;
        let transaction = self.with_reservation(reservation_id, |manager| {
            let process = start_runtime_with_cold_start_policy(
                &backend.id,
                &plan,
                RuntimeProcessOptions {
                    log_path,
                    env: runtime_env,
                    startup_timeout,
                    health_host: backend_host.clone(),
                },
                startup_cancelled,
            )?;
            local_state::save_selected_backend(&backend.id)?;
            local_state::save_selected_model(
                &resolved_model.model_path,
                mmproj_path.as_deref(),
                plan.ctx_size,
            )?;
            let generation = manager.take_generation()?;
            let allocation_id = manager
                .resource_ledger
                .as_mut()
                .expect("reservation requires a resource ledger")
                .commit(reservation_id)?;
            Ok((process, generation, allocation_id))
        })?;
        let (process, generation, allocation_id) = transaction;
        self.selected_backend = Some(backend.id.clone());
        self.loaded.insert(
            requested_model_key.clone(),
            LoadedRustRuntime {
                model_key: requested_model_key.clone(),
                owner_admin_id: owner_admin_id.clone(),
                backend_id: backend.id.clone(),
                model: resolved_model.model_path.clone(),
                public_model_id: public_model_id.clone(),
                mmproj: mmproj_path.clone(),
                ctx_size: plan.ctx_size,
                launch_args: effective_launch_args,
                cuda_visible_devices: cuda_selection
                    .as_ref()
                    .map(|selection| selection.visible_devices.clone()),
                cuda_warning: cuda_selection
                    .as_ref()
                    .and_then(|selection| selection.warning.clone()),
                external_server_protocol: plan.protocol,
                client_endpoint: plan.client_endpoint.clone(),
                proxy_model_ref: plan.proxy_model_ref.clone(),
                process,
                generation,
                route_state: RuntimeRouteState::Ready,
                allocation_id,
                resource_budget: resource_budget.clone(),
            },
        );
        self.default_model_key = Some(requested_model_key.clone());
        let loaded = self
            .loaded
            .get(&requested_model_key)
            .expect("newly loaded runtime should be registered");
        Ok(LoadModelOutcome::Success(model_load_response(
            loaded, false,
        )))
    }

    pub(super) fn unload_model(&mut self, model: &str, admin_id: Option<&str>) -> Result<Value> {
        let model_key = self
            .resolve_loaded_model_key(model)
            .ok_or_else(|| anyhow::anyhow!("model is not loaded: {model}"))?;
        let owner = self
            .loaded
            .get(&model_key)
            .and_then(|runtime| runtime.owner_admin_id.as_deref())
            .map(str::to_string);
        if let Some(owner) = owner.as_deref()
            && let Some(admin_id) = admin_id
            && owner != admin_id
        {
            anyhow::bail!(
                "model '{model_key}' is owned by admin '{owner}' and cannot be unloaded by admin '{admin_id}'"
            );
        }
        let (generation, stop_result) = {
            let loaded = self
                .loaded
                .get_mut(&model_key)
                .expect("resolved runtime key must exist");
            loaded.route_state = RuntimeRouteState::Draining;
            (
                loaded.generation,
                loaded.process.stop(Duration::from_secs(8)),
            )
        };
        if let Err(error) = stop_result {
            if let Some(loaded) = self.loaded.get_mut(&model_key) {
                loaded.route_state = RuntimeRouteState::Failed;
            }
            return Err(error.into());
        }
        self.remove_runtime_and_release(&model_key);
        self.select_fallback_default();
        Ok(json!({
            "ok": true,
            "unloaded": true,
            "model": model_key,
            "owner_admin_id": owner,
            "invalidated_generation": generation,
            "resources_released": true,
        }))
    }

    pub(super) fn resolve_requested_backend(&self, payload: &Value) -> Result<String> {
        payload
            .get("backend")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .or_else(|| self.selected_backend.clone())
            .or_else(|| {
                BackendRegistry::load_current()
                    .api_payload(BackendScope::Installed)
                    .get("recommended")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .ok_or_else(|| anyhow::anyhow!("no installed backend available"))
    }

    pub(super) fn proxy_base_for_model(&mut self, requested_model: Option<&str>) -> Option<String> {
        self.proxy_target_for_model(requested_model)
            .and_then(|target| target.base_url)
    }

    pub(super) fn proxy_target_for_model(
        &mut self,
        requested_model: Option<&str>,
    ) -> Option<RuntimeProxyTarget> {
        self.reap_exited_runtimes();
        let key = self.resolve_proxy_model_key(requested_model)?;
        let loaded = self.loaded.get(&key)?;
        if loaded.route_state != RuntimeRouteState::Ready {
            return None;
        }
        Some(RuntimeProxyTarget {
            base_url: loaded
                .external_server_protocol
                .is_openai_compatible()
                .then(|| loaded.client_endpoint.clone()),
            client_endpoint: loaded.client_endpoint.clone(),
            protocol: loaded.external_server_protocol,
            backend_id: loaded.backend_id.clone(),
            model: loaded.proxy_model_ref.clone(),
            generation: loaded.generation,
        })
    }

    fn resolve_proxy_model_key(&self, requested_model: Option<&str>) -> Option<String> {
        match requested_model
            .map(str::trim)
            .filter(|model| !model.is_empty())
        {
            Some("omniinfer" | "local") => self.default_model_key.clone(),
            Some(model) => self.resolve_loaded_model_key(model),
            None => self.default_model_key.clone(),
        }
    }

    fn resolve_loaded_model_key(&self, requested: &str) -> Option<String> {
        let requested = requested.trim();
        if requested.is_empty() {
            return None;
        }
        if self.loaded.contains_key(requested) {
            return Some(requested.to_string());
        }
        self.loaded.iter().find_map(|(key, loaded)| {
            (loaded.public_model_id.as_deref() == Some(requested)
                || loaded.model == requested
                || loaded.proxy_model_ref.as_deref() == Some(requested))
            .then(|| key.clone())
        })
    }

    fn matching_loaded_model_key(
        &self,
        requested_key: &str,
        model_path: &str,
        public_model_id: Option<&str>,
    ) -> Option<String> {
        if self.loaded.contains_key(requested_key) {
            return Some(requested_key.to_string());
        }
        self.loaded.iter().find_map(|(key, loaded)| {
            let compatible_public_id = loaded.public_model_id.is_none()
                || public_model_id.is_none()
                || loaded.public_model_id.as_deref() == public_model_id;
            (loaded.model == model_path && compatible_public_id).then(|| key.clone())
        })
    }

    fn promote_loaded_model_key(
        &mut self,
        loaded_key: &str,
        requested_key: &str,
        public_model_id: Option<&str>,
    ) -> String {
        if loaded_key == requested_key || public_model_id.is_none() {
            return loaded_key.to_string();
        }
        let Some(mut loaded) = self.loaded.remove(loaded_key) else {
            return loaded_key.to_string();
        };
        if loaded.public_model_id.is_some() {
            self.loaded.insert(loaded_key.to_string(), loaded);
            return loaded_key.to_string();
        }
        loaded.model_key = requested_key.to_string();
        loaded.public_model_id = public_model_id.map(str::to_string);
        self.loaded.insert(requested_key.to_string(), loaded);
        requested_key.to_string()
    }

    pub(super) fn loaded_models_payload(&mut self) -> Value {
        self.reap_exited_runtimes();
        json!({
            "object": "list",
            "data": self.loaded.values().map(loaded_runtime_payload).collect::<Vec<_>>(),
        })
    }

    pub(super) fn loaded_runtime_summaries(&mut self) -> Vec<LoadedRuntimeSummary> {
        self.reap_exited_runtimes();
        self.loaded
            .values()
            .filter(|loaded| loaded.route_state == RuntimeRouteState::Ready)
            .map(|loaded| LoadedRuntimeSummary {
                id: loaded.model_key.clone(),
                owner_admin_id: loaded.owner_admin_id.clone(),
                backend_pid: loaded.process.info().pid,
            })
            .collect()
    }

    pub(super) fn snapshot(&mut self) -> Value {
        self.reap_exited_runtimes();
        let persistent_state = local_state::load_state().unwrap_or_default();
        let selected_backend = self
            .selected_backend
            .clone()
            .or_else(|| persistent_state.selected_backend.clone());
        let loaded_models = self
            .loaded
            .values()
            .map(loaded_runtime_payload)
            .collect::<Vec<_>>();
        let mut payload = match self
            .default_model_key
            .as_ref()
            .and_then(|default_key| self.loaded.get(default_key))
        {
            Some(loaded) => {
                let info = loaded.process.info();
                json!({
                    "backend": loaded.backend_id,
                    "backend_ready": true,
                    "model": loaded.model_key,
                    "model_path": loaded.model,
                    "public_model_id": loaded.public_model_id,
                    "owner_admin_id": loaded.owner_admin_id,
                    "mmproj": loaded.mmproj,
                    "ctx_size": loaded.ctx_size,
                    "request_defaults": {},
                    "runtime_mode": "external_server",
                    "backend_pid": info.pid,
                    "backend_port": info.port,
                    "generation": loaded.generation,
                    "route_state": loaded.route_state.as_str(),
                    "allocation_id": loaded.allocation_id.get(),
                    "resource_budget": resource_budget_payload(&loaded.resource_budget),
                    "launch_args": loaded.launch_args,
                    "cuda_visible_devices": loaded.cuda_visible_devices,
                    "warning": loaded.cuda_warning,
                    "launch_command": info.command,
                    "proxy_model": loaded.proxy_model_ref,
                    "external_server_protocol": loaded.external_server_protocol.as_str(),
                    "client_endpoint": loaded.client_endpoint,
                    "openai_compatible": loaded.external_server_protocol.is_openai_compatible(),
                    "backend_log": info.log_path.display().to_string(),
                    "effective_parameters": {},
                    "runtime": {
                        "mode": "external_server",
                        "host": "127.0.0.1",
                        "port": info.port,
                        "pid": info.pid,
                        "cuda_visible_devices": loaded.cuda_visible_devices,
                        "launch_command": info.command,
                        "log_path": info.log_path.display().to_string(),
                        "proxy_model_ref": loaded.proxy_model_ref,
                        "external_server_protocol": loaded.external_server_protocol.as_str(),
                        "client_endpoint": loaded.client_endpoint,
                        "openai_compatible": loaded.external_server_protocol.is_openai_compatible(),
                    },
                    "log_path": info.log_path.display().to_string(),
                    "loaded_models": loaded_models,
                    "default_model": loaded.model_key,
                })
            }
            None => json!({
                "backend": selected_backend,
                "backend_ready": false,
                "model": null,
                "public_model_id": null,
                "mmproj": null,
                "ctx_size": null,
                "request_defaults": {},
                "runtime_mode": null,
                "backend_pid": null,
                "backend_port": null,
                "launch_args": [],
                "cuda_visible_devices": null,
                "warning": null,
                "launch_command": [],
                "proxy_model": null,
                "external_server_protocol": null,
                "client_endpoint": null,
                "openai_compatible": true,
                "backend_log": null,
                "effective_parameters": {},
                "runtime": null,
                "loaded_models": loaded_models,
                "default_model": null,
            }),
        };
        annotate_restore_state(&mut payload, &persistent_state, &self.loaded);
        payload["resource_ledger"] = self.resource_ledger_payload();
        payload
    }

    fn with_reservation<T>(
        &mut self,
        reservation_id: ReservationId,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let result = operation(self);
        if result.is_err() {
            self.resource_ledger
                .as_mut()
                .expect("reservation requires a resource ledger")
                .rollback(reservation_id);
        }
        result
    }

    fn take_generation(&mut self) -> Result<u64> {
        let generation = self.next_generation;
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("runtime generation overflow"))?;
        Ok(generation)
    }

    fn reserve_runtime_resources(
        &mut self,
        request_id: &str,
        budget: &ResourceBudget,
        cuda_visible_devices: Option<&str>,
    ) -> Result<ReservationId> {
        let observed = detect_available_resources(cuda_visible_devices)?;
        let current_usage = self
            .resource_ledger
            .as_ref()
            .map(|ledger| ledger.snapshot())
            .map(|snapshot| {
                merge_domain_totals(&snapshot.reserved, &snapshot.committed)
                    .map(|usage| (snapshot, usage))
            })
            .transpose()?;
        let mut capacities = current_usage
            .as_ref()
            .map(|(snapshot, _)| snapshot.capacities.clone())
            .unwrap_or_default();
        for (domain, available) in observed {
            let used = current_usage
                .as_ref()
                .and_then(|(_, usage)| usage.get(&domain))
                .copied()
                .unwrap_or(0);
            capacities.insert(
                domain,
                available
                    .checked_add(used)
                    .ok_or_else(|| anyhow::anyhow!("resource capacity overflow"))?,
            );
        }
        let snapshot_id = self.next_capacity_snapshot;
        self.next_capacity_snapshot = self
            .next_capacity_snapshot
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("resource capacity snapshot overflow"))?;
        let capacity = ResourceCapacity::new(snapshot_id, capacities)?;
        match self.resource_ledger.as_mut() {
            Some(ledger) => ledger.update_capacity(capacity)?,
            None => self.resource_ledger = Some(ResourceLedger::new(capacity)),
        }
        Ok(self
            .resource_ledger
            .as_mut()
            .expect("resource ledger was initialized")
            .reserve(request_id, budget.clone())?)
    }

    fn reap_exited_runtimes(&mut self) {
        let exited = self
            .loaded
            .iter_mut()
            .filter_map(|(key, loaded)| match loaded.process.has_exited() {
                Ok(true) => Some(key.clone()),
                Ok(false) => None,
                Err(_) => {
                    loaded.route_state = RuntimeRouteState::Failed;
                    None
                }
            })
            .collect::<Vec<_>>();
        for key in exited {
            self.remove_runtime_and_release(&key);
        }
        self.select_fallback_default();
    }

    fn remove_runtime_and_release(&mut self, key: &str) {
        if let Some(loaded) = self.loaded.remove(key)
            && let Some(ledger) = self.resource_ledger.as_mut()
        {
            ledger.release(loaded.allocation_id);
        }
    }

    fn select_fallback_default(&mut self) {
        if self.default_model_key.as_ref().is_some_and(|key| {
            self.loaded
                .get(key)
                .is_some_and(|loaded| loaded.route_state == RuntimeRouteState::Ready)
        }) {
            return;
        }
        self.default_model_key = self.loaded.iter().rev().find_map(|(key, loaded)| {
            (loaded.route_state == RuntimeRouteState::Ready).then(|| key.clone())
        });
    }

    fn resource_ledger_payload(&self) -> Value {
        let Some(ledger) = self.resource_ledger.as_ref() else {
            return Value::Null;
        };
        let snapshot = ledger.snapshot();
        let available = snapshot.available().unwrap_or_default();
        json!({
            "capacity_snapshot_id": snapshot.capacity_snapshot_id,
            "capacity_bytes": domain_bytes_payload(&snapshot.capacities),
            "reserved_bytes": domain_bytes_payload(&snapshot.reserved),
            "committed_bytes": domain_bytes_payload(&snapshot.committed),
            "available_bytes": domain_bytes_payload(&available),
        })
    }
}

fn start_runtime_with_cold_start_policy(
    backend_id: &str,
    plan: &omniinfer_core::runtime_plan::ExternalRuntimePlan,
    options: RuntimeProcessOptions,
    startup_cancelled: &AtomicBool,
) -> Result<RuntimeProcess, RuntimeProcessError> {
    let Some(initial_timeout) =
        wsl_rocm_cold_start_retry_timeout(backend_id, options.startup_timeout)
    else {
        return RuntimeProcess::start_cancellable(plan, options, startup_cancelled);
    };

    let total_timeout = options.startup_timeout;
    retry_after_ready_timeout(
        total_timeout,
        initial_timeout,
        WSL_ROCM_COLD_START_RETRY_COOLDOWN,
        startup_cancelled,
        |attempt_timeout| {
            let mut attempt_options = options.clone();
            attempt_options.startup_timeout = attempt_timeout;
            RuntimeProcess::start_cancellable(plan, attempt_options, startup_cancelled)
        },
    )
}

fn wsl_rocm_cold_start_retry_timeout(
    backend_id: &str,
    total_timeout: Duration,
) -> Option<Duration> {
    (backend_id == "vllm-wsl2-rocm" && total_timeout >= WSL_ROCM_COLD_START_RETRY_MINIMUM_BUDGET)
        .then_some(WSL_ROCM_COLD_START_INITIAL_ATTEMPT)
}

fn retry_after_ready_timeout<T>(
    total_timeout: Duration,
    initial_timeout: Duration,
    cooldown: Duration,
    startup_cancelled: &AtomicBool,
    mut attempt: impl FnMut(Duration) -> Result<T, RuntimeProcessError>,
) -> Result<T, RuntimeProcessError> {
    let started = Instant::now();
    match attempt(initial_timeout) {
        Err(RuntimeProcessError::ReadyTimeout) => {
            let remaining_before_cooldown = total_timeout.saturating_sub(started.elapsed());
            if remaining_before_cooldown <= cooldown {
                return Err(RuntimeProcessError::ReadyTimeout);
            }
            eprintln!(
                "OmniInfer: WSL2 ROCm cold start did not become ready after {} seconds; cooling down for {} seconds before retry",
                initial_timeout.as_secs(),
                cooldown.as_secs()
            );
            let cooldown_deadline = Instant::now() + cooldown;
            while Instant::now() < cooldown_deadline {
                if startup_cancelled.load(Ordering::SeqCst) {
                    return Err(RuntimeProcessError::Interrupted);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let remaining = total_timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(RuntimeProcessError::ReadyTimeout);
            }
            eprintln!(
                "OmniInfer: retrying WSL2 ROCm cold start once with the remaining {} seconds",
                remaining.as_secs()
            );
            attempt(remaining)
        }
        result => result,
    }
}

fn annotate_restore_state(
    payload: &mut Value,
    persistent_state: &local_state::LocalState,
    loaded_runtimes: &BTreeMap<String, LoadedRustRuntime>,
) {
    let Some(selected) = persistent_state.selected_model.as_ref() else {
        payload["restore_selection"] = Value::Null;
        payload["restore_status"] = json!("not_configured");
        payload["restore_completed"] = json!(false);
        return;
    };
    let completed = loaded_runtimes.values().any(|loaded| {
        loaded.route_state == RuntimeRouteState::Ready
            && persistent_state
                .selected_backend
                .as_deref()
                .is_none_or(|backend| loaded.backend_id == backend)
            && loaded.model == selected.model
            && loaded.mmproj == selected.mmproj
            && loaded.ctx_size == selected.ctx_size
    });
    payload["restore_selection"] = json!({
        "backend": persistent_state.selected_backend,
        "model": selected.model,
        "mmproj": selected.mmproj,
        "ctx_size": selected.ctx_size,
    });
    payload["restore_status"] = json!(if completed { "loaded" } else { "pending" });
    payload["restore_completed"] = json!(completed);
}

fn same_load_configuration(
    loaded: &LoadedRustRuntime,
    backend_id: &str,
    model_path: &str,
    mmproj: Option<&str>,
    ctx_size: Option<u32>,
    launch_args: &[String],
) -> bool {
    loaded.backend_id == backend_id
        && loaded.model == model_path
        && loaded.mmproj.as_deref() == mmproj
        && loaded.ctx_size == ctx_size
        && loaded.launch_args == launch_args
}

fn model_load_response(loaded: &LoadedRustRuntime, already_loaded: bool) -> Value {
    let info = loaded.process.info();
    let mut response = json!({
        "ok": true,
        "already_loaded": already_loaded,
        "requires_reload": false,
        "model": loaded.model_key,
        "owner_admin_id": loaded.owner_admin_id,
        "selected_backend": loaded.backend_id,
        "selected_model": loaded.model,
        "selected_public_model_id": loaded.public_model_id,
        "selected_mmproj": loaded.mmproj,
        "selected_ctx_size": loaded.ctx_size,
        "backend_pid": info.pid,
        "backend_port": info.port,
        "generation": loaded.generation,
        "route_state": loaded.route_state.as_str(),
        "allocation_id": loaded.allocation_id.get(),
        "resource_budget": resource_budget_payload(&loaded.resource_budget),
        "launch_command": info.command,
        "log_path": info.log_path.display().to_string(),
        "external_server_protocol": loaded.external_server_protocol.as_str(),
        "client_endpoint": loaded.client_endpoint,
        "openai_compatible": loaded.external_server_protocol.is_openai_compatible(),
    });
    if let Some(visible_devices) = loaded.cuda_visible_devices.as_deref() {
        response["cuda_visible_devices"] = json!(visible_devices);
    }
    if let Some(warning) = loaded.cuda_warning.as_deref() {
        response["warning"] = json!(warning);
    }
    response
}

struct RequestedRuntimeConfig<'a> {
    backend_id: &'a str,
    model_key: &'a str,
    model_path: &'a str,
    public_model_id: Option<&'a str>,
    mmproj: Option<&'a str>,
    ctx_size: Option<u32>,
    launch_args: &'a [String],
}

fn reload_required_response(
    loaded: &LoadedRustRuntime,
    requested: &RequestedRuntimeConfig<'_>,
) -> Value {
    json!({
        "ok": false,
        "already_loaded": true,
        "requires_reload": true,
        "error": {
            "code": "model_reload_required",
            "message": format!(
                "model '{}' is already loaded with different runtime settings; unload it before selecting the new configuration",
                requested.model_key,
            ),
        },
        "current": {
            "backend": loaded.backend_id,
            "model": loaded.model_key,
            "model_path": loaded.model,
            "public_model_id": loaded.public_model_id,
            "mmproj": loaded.mmproj,
            "ctx_size": loaded.ctx_size,
            "launch_args": loaded.launch_args,
        },
        "requested": {
            "backend": requested.backend_id,
            "model": requested.model_key,
            "model_path": requested.model_path,
            "public_model_id": requested.public_model_id,
            "mmproj": requested.mmproj,
            "ctx_size": requested.ctx_size,
            "launch_args": requested.launch_args,
        },
    })
}

fn loaded_runtime_payload(loaded: &LoadedRustRuntime) -> Value {
    let info = loaded.process.info();
    json!({
        "id": loaded.model_key,
        "owner_admin_id": loaded.owner_admin_id,
        "backend": loaded.backend_id,
        "model": loaded.model_key,
        "model_path": loaded.model,
        "public_model_id": loaded.public_model_id,
        "mmproj": loaded.mmproj,
        "ctx_size": loaded.ctx_size,
        "runtime_mode": "external_server",
        "backend_pid": info.pid,
        "backend_port": info.port,
        "generation": loaded.generation,
        "route_state": loaded.route_state.as_str(),
        "allocation_id": loaded.allocation_id.get(),
        "resource_budget": resource_budget_payload(&loaded.resource_budget),
        "launch_args": loaded.launch_args,
        "cuda_visible_devices": loaded.cuda_visible_devices,
        "warning": loaded.cuda_warning,
        "launch_command": info.command,
        "proxy_model": loaded.proxy_model_ref,
        "external_server_protocol": loaded.external_server_protocol.as_str(),
        "client_endpoint": loaded.client_endpoint,
        "openai_compatible": loaded.external_server_protocol.is_openai_compatible(),
        "backend_log": info.log_path.display().to_string(),
    })
}

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

fn build_runtime_resource_budget(
    payload: &Value,
    backend: &backend_registry::BackendSpec,
    model: &str,
    mmproj: Option<&str>,
    ctx_size: u32,
    cuda_visible_devices: Option<&str>,
    replicate_across_domains: bool,
) -> Result<ResourceBudget> {
    let domains = if cfg!(target_os = "macos")
        || backend
            .capabilities
            .iter()
            .any(|value| value == "shared-memory")
    {
        vec![MemoryDomain::Unified("system".to_string())]
    } else if backend.capabilities.iter().any(|value| value == "cuda") {
        let devices = parse_cuda_devices(cuda_visible_devices.ok_or_else(|| {
            anyhow::anyhow!("CUDA resource budgeting requires a selected device")
        })?);
        if devices.is_empty() {
            anyhow::bail!("CUDA resource budgeting requires a selected device");
        }
        devices.into_iter().map(MemoryDomain::Cuda).collect()
    } else {
        vec![MemoryDomain::Host]
    };
    let explicit_total = payload
        .get("resource_budget_bytes")
        .and_then(Value::as_u64)
        .filter(|bytes| *bytes > 0);
    let weights = artifact_size_bytes(&PathBuf::from(model))?;
    let projector = mmproj
        .map(|path| artifact_size_bytes(&PathBuf::from(path)))
        .transpose()?
        .flatten()
        .unwrap_or(0);
    let Some(weights) = weights else {
        let total = explicit_total.ok_or_else(|| {
            anyhow::anyhow!(
                "model artifact size is unknown; provide a non-zero resource_budget_bytes value"
            )
        })?;
        return Ok(ResourceBudget::from_components(assign_component(
            "client_provided_total",
            total,
            &domains,
            replicate_across_domains,
        )?)?);
    };
    let weights = weights.max(1);
    let base = weights
        .checked_add(projector)
        .ok_or_else(|| anyhow::anyhow!("model artifact size overflow"))?;
    let parameter_proxy = base.saturating_mul(2).max(GIB);
    let ctx = u64::from(ctx_size.max(1));
    let kv_cache = checked_scaled(parameter_proxy, 3, 100)?
        .checked_mul(ctx)
        .and_then(|bytes| bytes.checked_div(u64::from(DEFAULT_LOAD_CONTEXT_SIZE)))
        .unwrap_or(u64::MAX)
        .max(256 * MIB);
    let activation_ctx = ctx.min(u64::from(DEFAULT_LOAD_CONTEXT_SIZE) * 4);
    let activation = checked_scaled(parameter_proxy, 1, 100)?
        .checked_mul(activation_ctx)
        .and_then(|bytes| bytes.checked_div(u64::from(DEFAULT_LOAD_CONTEXT_SIZE)))
        .unwrap_or(u64::MAX)
        .max(128 * MIB);
    let framework = checked_scaled(base, 8, 100)?.max(384 * MIB);
    let allocator_slack = checked_scaled(base, 4, 100)?.max(160 * MIB);
    let mut components = Vec::new();
    for (name, bytes) in [
        ("weights", weights),
        ("kv_cache", kv_cache),
        ("activation", activation),
        ("framework_overhead", framework),
        ("allocator_slack", allocator_slack),
    ] {
        components.extend(assign_component(
            name,
            bytes,
            &domains,
            replicate_across_domains,
        )?);
    }
    if projector > 0 {
        components.extend(assign_component(
            "mmproj",
            projector,
            &domains,
            replicate_across_domains,
        )?);
    }
    let estimated = ResourceBudget::from_components(components)?;
    if let Some(explicit_total) = explicit_total {
        let estimated_minimum = if replicate_across_domains {
            estimated.domains().values().copied().max().unwrap_or(0)
        } else {
            estimated
                .domains()
                .values()
                .try_fold(0_u64, |total, bytes| total.checked_add(*bytes))
                .ok_or_else(|| anyhow::anyhow!("resource budget overflow"))?
        };
        if explicit_total < estimated_minimum {
            anyhow::bail!(
                "resource_budget_bytes is below the estimated minimum of {estimated_minimum} bytes"
            );
        }
        return Ok(ResourceBudget::from_components(assign_component(
            "client_provided_total",
            explicit_total,
            &domains,
            replicate_across_domains,
        )?)?);
    }
    Ok(estimated)
}

fn artifact_size_bytes(path: &PathBuf) -> Result<Option<u64>> {
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(None);
    };
    if metadata.is_file() {
        return Ok(Some(metadata.len()));
    }
    if !metadata.is_dir() {
        return Ok(None);
    }
    let mut total = 0_u64;
    let mut pending = vec![path.clone()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file() {
                total = total
                    .checked_add(entry.metadata()?.len())
                    .ok_or_else(|| anyhow::anyhow!("model artifact size overflow"))?;
            }
        }
    }
    Ok((total > 0).then_some(total))
}

fn checked_scaled(value: u64, numerator: u64, denominator: u64) -> Result<u64> {
    value
        .checked_mul(numerator)
        .and_then(|scaled| scaled.checked_div(denominator))
        .ok_or_else(|| anyhow::anyhow!("resource budget overflow"))
}

fn assign_component(
    name: &str,
    bytes: u64,
    domains: &[MemoryDomain],
    replicate_across_domains: bool,
) -> Result<Vec<BudgetComponent>> {
    if replicate_across_domains {
        if domains.is_empty() || bytes == 0 {
            anyhow::bail!("resource component requires non-zero bytes and at least one domain");
        }
        return Ok(domains
            .iter()
            .map(|domain| BudgetComponent {
                name: name.to_string(),
                domain: domain.clone(),
                bytes,
            })
            .collect());
    }
    distribute_component(name, bytes, domains)
}

fn distribute_component(
    name: &str,
    bytes: u64,
    domains: &[MemoryDomain],
) -> Result<Vec<BudgetComponent>> {
    let count = u64::try_from(domains.len())?;
    if count == 0 || bytes == 0 {
        anyhow::bail!("resource component requires non-zero bytes and at least one domain");
    }
    let base = bytes / count;
    let remainder = bytes % count;
    domains
        .iter()
        .enumerate()
        .map(|(index, domain)| {
            let bytes = base + u64::from(u64::try_from(index)? < remainder);
            Ok(BudgetComponent {
                name: name.to_string(),
                domain: domain.clone(),
                bytes,
            })
        })
        .collect()
}

fn parse_cuda_devices(visible_devices: &str) -> Vec<String> {
    let mut devices = visible_devices
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    devices.sort();
    devices.dedup();
    devices
}

fn detect_available_resources(
    cuda_visible_devices: Option<&str>,
) -> Result<BTreeMap<MemoryDomain, u64>> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available_memory = system.available_memory();
    if available_memory == 0 {
        anyhow::bail!("available system memory could not be detected");
    }
    let mut domains = BTreeMap::from([
        (MemoryDomain::Host, available_memory),
        (
            MemoryDomain::Unified("system".to_string()),
            available_memory,
        ),
    ]);
    if let Some(devices) = cuda_visible_devices {
        domains.extend(cuda_available_bytes(devices)?);
    }
    Ok(domains)
}

#[cfg(test)]
fn detect_cuda_device_ids() -> Result<Vec<String>> {
    Ok(vec!["0".to_string()])
}

#[cfg(not(test))]
fn detect_cuda_device_ids() -> Result<Vec<String>> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader,nounits"])
        .output()?;
    if !output.status.success() {
        anyhow::bail!("nvidia-smi device query failed");
    }
    let devices = parse_cuda_devices(&String::from_utf8_lossy(&output.stdout).replace('\n', ","));
    if devices.is_empty() {
        anyhow::bail!("nvidia-smi did not report any CUDA devices");
    }
    Ok(devices)
}

#[cfg(test)]
fn cuda_available_bytes(visible_devices: &str) -> Result<BTreeMap<MemoryDomain, u64>> {
    const TEST_CUDA_CAPACITY: u64 = 1024 * GIB;
    let requested = parse_cuda_devices(visible_devices);
    if requested.is_empty() {
        anyhow::bail!("CUDA device selection is empty");
    }
    Ok(requested
        .into_iter()
        .map(|device| (MemoryDomain::Cuda(device), TEST_CUDA_CAPACITY))
        .collect())
}

#[cfg(not(test))]
fn cuda_available_bytes(visible_devices: &str) -> Result<BTreeMap<MemoryDomain, u64>> {
    let requested = parse_cuda_devices(visible_devices);
    if requested.is_empty() {
        anyhow::bail!("CUDA device selection is empty");
    }
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        anyhow::bail!("nvidia-smi memory query failed");
    }
    let rows = String::from_utf8_lossy(&output.stdout);
    let mut available = BTreeMap::new();
    for requested_device in &requested {
        let memory_mib = rows.lines().find_map(|line| {
            let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
            (parts.len() >= 3 && (parts[0] == *requested_device || parts[1] == *requested_device))
                .then(|| parts[2].parse::<u64>().ok())
                .flatten()
        });
        let memory_mib = memory_mib.ok_or_else(|| {
            anyhow::anyhow!(
                "selected CUDA device was not reported by nvidia-smi: {requested_device}"
            )
        })?;
        available.insert(
            MemoryDomain::Cuda(requested_device.clone()),
            memory_mib
                .checked_mul(MIB)
                .ok_or_else(|| anyhow::anyhow!("CUDA capacity overflow"))?,
        );
    }
    Ok(available)
}

fn merge_domain_totals(
    left: &BTreeMap<MemoryDomain, u64>,
    right: &BTreeMap<MemoryDomain, u64>,
) -> Result<BTreeMap<MemoryDomain, u64>> {
    let mut merged = left.clone();
    for (domain, bytes) in right {
        let total = merged.entry(domain.clone()).or_insert(0);
        *total = total
            .checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("resource usage overflow"))?;
    }
    Ok(merged)
}

fn domain_bytes_payload(domains: &BTreeMap<MemoryDomain, u64>) -> Value {
    Value::Object(
        domains
            .iter()
            .map(|(domain, bytes)| (domain.key(), json!(bytes)))
            .collect(),
    )
}

fn resource_budget_payload(budget: &ResourceBudget) -> Value {
    json!({
        "domains_bytes": domain_bytes_payload(budget.domains()),
        "components": budget.components().iter().map(|component| json!({
            "name": component.name,
            "domain": component.domain.key(),
            "bytes": component.bytes,
        })).collect::<Vec<_>>(),
    })
}

fn model_log_file_name(base: &str, model_key: &str) -> String {
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

fn json_required_str<'a>(payload: &'a Value, key: &'static str) -> Result<&'a str> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("field '{key}' is required"))
}

fn resolve_model_for_backend(
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

fn is_vla_checkpoint_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("gguf") || extension.eq_ignore_ascii_case("safetensors")
        })
}

fn resolve_path_for_backend(
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

fn expand_home(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path
}

fn launch_args_have_ctx_size(family: &str, args: &[String]) -> bool {
    args.iter().any(|arg| {
        let flag = arg.split_once('=').map(|(flag, _)| flag).unwrap_or(arg);
        match family {
            "vllm" => flag == "--max-model-len",
            "llama.cpp" | "turboquant" => matches!(flag, "-c" | "--ctx-size"),
            _ => matches!(flag, "-c" | "--ctx-size" | "--max-model-len"),
        }
    })
}

fn merged_launch_args(
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

pub(super) fn pick_runtime_port(host: &str) -> Result<u16> {
    let listener = std::net::TcpListener::bind((host, 0))?;
    Ok(listener.local_addr()?.port())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_budget(bytes: u64) -> ResourceBudget {
        ResourceBudget::from_domains(BTreeMap::from([(MemoryDomain::Host, bytes)])).unwrap()
    }

    #[test]
    fn detects_llama_context_args() {
        assert!(launch_args_have_ctx_size(
            "llama.cpp",
            &["-c".to_string(), "8192".to_string()]
        ));
        assert!(launch_args_have_ctx_size(
            "llama.cpp",
            &["--ctx-size=4096".to_string()]
        ));
        assert!(!launch_args_have_ctx_size(
            "llama.cpp",
            &["-ngl".to_string(), "999".to_string()]
        ));
    }

    #[test]
    fn detects_vllm_context_args() {
        assert!(launch_args_have_ctx_size(
            "vllm",
            &["--max-model-len=65536".to_string()]
        ));
        assert!(!launch_args_have_ctx_size(
            "vllm",
            &["--gpu-memory-utilization".to_string(), "0.9".to_string()]
        ));
    }

    #[test]
    fn failed_load_transaction_rolls_back_reservation() {
        let mut manager = RustRuntimeManager {
            resource_ledger: Some(ResourceLedger::new(
                ResourceCapacity::new(1, BTreeMap::from([(MemoryDomain::Host, 1024)])).unwrap(),
            )),
            ..Default::default()
        };
        let reservation = manager
            .resource_ledger
            .as_mut()
            .unwrap()
            .reserve("failed-load", test_budget(768))
            .unwrap();

        let result: Result<()> = manager.with_reservation(reservation, |_| {
            Err(anyhow::anyhow!("simulated readiness timeout"))
        });

        assert!(result.is_err());
        let snapshot = manager.resource_ledger.as_ref().unwrap().snapshot();
        assert!(snapshot.reserved.is_empty());
        assert!(snapshot.committed.is_empty());
    }

    #[test]
    fn multi_gpu_components_are_split_into_non_overlapping_domains() {
        let domains = vec![
            MemoryDomain::Cuda("0".to_string()),
            MemoryDomain::Cuda("1".to_string()),
        ];
        let components = distribute_component("weights", 101, &domains).unwrap();
        let budget = ResourceBudget::from_components(components).unwrap();

        assert_eq!(budget.domains()[&MemoryDomain::Cuda("0".to_string())], 51);
        assert_eq!(budget.domains()[&MemoryDomain::Cuda("1".to_string())], 50);
        assert!(
            !budget
                .domains()
                .contains_key(&MemoryDomain::Cuda("0,1".to_string()))
        );
    }

    #[test]
    fn uncertain_multi_gpu_mapping_reserves_full_budget_per_device() {
        let domains = vec![
            MemoryDomain::Cuda("0".to_string()),
            MemoryDomain::Cuda("1".to_string()),
        ];
        let components = assign_component("weights", 101, &domains, true).unwrap();
        let budget = ResourceBudget::from_components(components).unwrap();

        assert_eq!(budget.domains()[&MemoryDomain::Cuda("0".to_string())], 101);
        assert_eq!(budget.domains()[&MemoryDomain::Cuda("1".to_string())], 101);
    }

    #[test]
    fn explicit_budget_cannot_understate_local_estimate() {
        let root = std::env::temp_dir().join(format!(
            "omniinfer-resource-budget-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        let model = root.join("model.gguf");
        fs::write(&model, vec![0_u8; 1024]).unwrap();
        let backend_id = if cfg!(target_os = "linux") {
            "llama.cpp-linux"
        } else if cfg!(target_os = "macos") {
            "llama.cpp-mac-intel"
        } else {
            "llama.cpp-cpu"
        };
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(backend_id)
            .expect("test platform should expose a CPU external backend");

        let result = build_runtime_resource_budget(
            &json!({"resource_budget_bytes": 1024}),
            backend,
            model.to_str().unwrap(),
            None,
            512,
            None,
            false,
        );

        assert!(result.is_err());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn recognizes_only_supported_vla_checkpoint_extensions() {
        assert!(is_vla_checkpoint_path(
            PathBuf::from("model.gguf").as_path()
        ));
        assert!(is_vla_checkpoint_path(
            PathBuf::from("model.SAFETENSORS").as_path()
        ));
        assert!(!is_vla_checkpoint_path(
            PathBuf::from("model.bin").as_path()
        ));
        assert!(!is_vla_checkpoint_path(PathBuf::from("model").as_path()));
    }

    #[test]
    fn official_llama_launch_args_extend_defaults_with_user_overrides_last() {
        let defaults = vec![
            "--slot-prompt-similarity".to_string(),
            "0".to_string(),
            "--cache-idle-slots".to_string(),
            "--cache-ram".to_string(),
            "8192".to_string(),
        ];
        let requested = vec![
            "-np".to_string(),
            "5".to_string(),
            "--cache-ram".to_string(),
            "32768".to_string(),
        ];

        assert_eq!(
            merged_launch_args(
                "llama.cpp-linux-cuda",
                "llama.cpp",
                &defaults,
                Some(&requested)
            ),
            vec![
                "--slot-prompt-similarity",
                "0",
                "--cache-idle-slots",
                "--cache-ram",
                "8192",
                "-np",
                "5",
                "--cache-ram",
                "32768"
            ]
        );
        assert_eq!(
            merged_launch_args("llama.cpp-linux-cuda", "llama.cpp", &defaults, None),
            defaults
        );
    }

    #[test]
    fn non_official_llama_launch_args_keep_replacement_semantics() {
        let defaults = vec!["--jinja".to_string(), "-ngl".to_string(), "999".to_string()];
        let requested = vec!["-ngl".to_string(), "12".to_string()];

        assert_eq!(
            merged_launch_args(
                "ik_llama.cpp-linux-cuda",
                "llama.cpp",
                &defaults,
                Some(&requested)
            ),
            requested
        );
    }

    #[test]
    fn wsl_rocm_cold_start_retry_requires_a_safe_total_budget() {
        assert_eq!(
            wsl_rocm_cold_start_retry_timeout("vllm-wsl2-rocm", Duration::from_secs(420)),
            Some(Duration::from_secs(120))
        );
        assert_eq!(
            wsl_rocm_cold_start_retry_timeout("vllm-wsl2-rocm", Duration::from_secs(359)),
            None
        );
        assert_eq!(
            wsl_rocm_cold_start_retry_timeout("vllm-wsl2-cuda", Duration::from_secs(420)),
            None
        );
    }

    #[test]
    fn ready_timeout_retries_once_with_the_remaining_budget() {
        let total_timeout = Duration::from_secs(300);
        let mut attempts = Vec::new();
        let cancelled = AtomicBool::new(false);
        let result = retry_after_ready_timeout(
            total_timeout,
            Duration::from_secs(120),
            Duration::ZERO,
            &cancelled,
            |timeout| {
                attempts.push(timeout);
                if attempts.len() == 1 {
                    Err(RuntimeProcessError::ReadyTimeout)
                } else {
                    Ok("ready")
                }
            },
        )
        .unwrap();

        assert_eq!(result, "ready");
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0], Duration::from_secs(120));
        assert!(attempts[1] <= total_timeout);
        assert!(attempts[1] >= Duration::from_secs(299));
    }

    #[test]
    fn cold_start_retry_does_not_mask_early_exit() {
        let mut attempts = 0;
        let cancelled = AtomicBool::new(false);
        let error = retry_after_ready_timeout(
            Duration::from_secs(300),
            Duration::from_secs(120),
            Duration::ZERO,
            &cancelled,
            |_| {
                attempts += 1;
                Err::<(), _>(RuntimeProcessError::EarlyExit)
            },
        )
        .unwrap_err();

        assert!(matches!(error, RuntimeProcessError::EarlyExit));
        assert_eq!(attempts, 1);
    }

    #[test]
    fn ready_timeout_does_not_retry_without_post_cooldown_budget() {
        let mut attempts = 0;
        let cancelled = AtomicBool::new(false);
        let error = retry_after_ready_timeout(
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::from_millis(1),
            &cancelled,
            |_| {
                attempts += 1;
                Err::<(), _>(RuntimeProcessError::ReadyTimeout)
            },
        )
        .unwrap_err();

        assert!(matches!(error, RuntimeProcessError::ReadyTimeout));
        assert_eq!(attempts, 1);
    }
}
