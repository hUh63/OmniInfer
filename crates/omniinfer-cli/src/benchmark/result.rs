use super::*;

pub(super) fn read_prompt(args: &BenchRunArgs) -> Result<(String, String)> {
    match args.prompt_file.as_deref() {
        Some(path) => {
            let raw = fs::read(path)
                .with_context(|| format!("failed to read prompt file {}", path.display()))?;
            let prompt = String::from_utf8(raw).context("prompt file must be UTF-8")?;
            if prompt.trim().is_empty() {
                anyhow::bail!("prompt file must not be empty.");
            }
            Ok((prompt, "OmniInfer prompt file".to_string()))
        }
        None => {
            let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT).to_string();
            if prompt.trim().is_empty() {
                anyhow::bail!("--prompt must not be empty.");
            }
            Ok((prompt, "OmniInfer inline prompt".to_string()))
        }
    }
}

pub(super) fn protocol_notes(args: &BenchRunArgs) -> Option<String> {
    match (args.notes.as_deref(), args.ignore_eos) {
        (Some(notes), true) => Some(format!("{notes}; {IGNORE_EOS_PROTOCOL_NOTE}")),
        (None, true) => Some(IGNORE_EOS_PROTOCOL_NOTE.to_string()),
        (Some(notes), false) => Some(notes.to_string()),
        (None, false) => None,
    }
}

pub(super) fn extract_measurement(response: &Value, elapsed: Duration) -> Result<Measurement> {
    let usage = response
        .get("usage")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("response is missing usage"))?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow::anyhow!("response is missing positive prompt_tokens"))?;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow::anyhow!("response is missing positive completion_tokens"))?;
    let timings = response.get("timings").and_then(Value::as_object);
    let metrics = response.get("omniinfer_metrics").and_then(Value::as_object);
    let prefill_duration_ms = positive_number(timings, &["prompt_ms"])
        .or_else(|| {
            positive_number(timings, &["prompt_per_second"])
                .map(|tps| prompt_tokens as f64 * 1000.0 / tps)
        })
        .or_else(|| {
            positive_number(metrics, &["observed_prefill_tps"])
                .map(|tps| prompt_tokens as f64 * 1000.0 / tps)
        })
        .or_else(|| positive_number(metrics, &["ttft_ms"]))
        .ok_or_else(|| anyhow::anyhow!("response has no prefill timing"))?;
    let decode_duration_ms = positive_number(timings, &["predicted_ms", "decode_ms"])
        .or_else(|| {
            positive_number(timings, &["predicted_per_second", "decode_tps"])
                .map(|tps| completion_tokens as f64 * 1000.0 / tps)
        })
        .or_else(|| {
            positive_number(metrics, &["observed_decode_tps"])
                .map(|tps| completion_tokens as f64 * 1000.0 / tps)
        })
        .or_else(|| positive_number(metrics, &["decode_ms"]))
        .ok_or_else(|| anyhow::anyhow!("response has no decode timing"))?;
    let ttft_ms = positive_number(metrics, &["ttft_ms"]);
    let wall_time_ms = elapsed.as_secs_f64() * 1000.0;
    Ok(Measurement {
        prompt_tokens,
        completion_tokens,
        prefill_tps: prompt_tokens as f64 * 1000.0 / prefill_duration_ms,
        decode_tps: completion_tokens as f64 * 1000.0 / decode_duration_ms,
        prefill_duration_ms,
        decode_duration_ms,
        ttft_ms,
        wall_time_ms,
    })
}

pub(super) fn positive_number(object: Option<&Map<String, Value>>, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| object?.get(*key)?.as_f64())
        .filter(|value| value.is_finite() && *value > 0.0)
}

pub(super) fn consistent_token_count(measurements: &[Measurement], prompt: bool) -> Result<u64> {
    let first = measurements
        .first()
        .map(|measurement| {
            if prompt {
                measurement.prompt_tokens
            } else {
                measurement.completion_tokens
            }
        })
        .ok_or_else(|| anyhow::anyhow!("benchmark produced no measurements"))?;
    if measurements.iter().any(|measurement| {
        let value = if prompt {
            measurement.prompt_tokens
        } else {
            measurement.completion_tokens
        };
        value != first
    }) {
        let label = if prompt { "prompt" } else { "completion" };
        anyhow::bail!(
            "Measured {label} token counts differ between runs; a single PP/TG result would be ambiguous."
        );
    }
    Ok(first)
}

pub(super) struct BuildSubmission<'a> {
    pub(super) args: &'a BenchRunArgs,
    pub(super) benchmark_id: &'a str,
    pub(super) loaded_backend: &'a str,
    pub(super) run_command: &'a str,
    pub(super) optimization_mode: &'a str,
    pub(super) optimizations: &'a [String],
    pub(super) context_size: u32,
    pub(super) batch_size: u32,
    pub(super) prompt_source: &'a str,
    pub(super) prompt_sha256: &'a str,
    pub(super) started_at: OffsetDateTime,
    pub(super) measurements: &'a [Measurement],
    pub(super) pp: u64,
    pub(super) tg: u64,
}

pub(super) fn build_submission(input: BuildSubmission<'_>) -> Result<Value> {
    let mut model = json!({
        "catalog_model_id": input.args.catalog_model_id,
        "format": input.args.model_format,
        "quantization": input.args.quantization,
        "download_url": input.args.model_url,
    });
    if let Some(name) = input.args.model_name.as_deref() {
        model["name"] = json!(name);
    }
    let mut backend = json!({
        "catalog_backend_id": input.loaded_backend,
        "version": input.args.backend_version,
    });
    if let Some(name) = input.args.backend_name.as_deref() {
        backend["name"] = json!(name);
    }
    let mut protocol = json!({
        "run_mode": "steady_state",
        "cache_policy": "reused_within_submission",
        "timeout_seconds": input.args.timeout_seconds,
        "started_at": input.started_at.format(&Rfc3339)?,
    });
    if let Some(notes) = protocol_notes(input.args) {
        protocol["notes"] = json!(notes);
    }
    let mut provenance = json!({"submitter_name": input.args.submitter_name});
    if let Some(organization) = input.args.organization.as_deref() {
        provenance["organization"] = json!(organization);
    }
    if let Some(source_url) = input.args.source_url.as_deref() {
        provenance["source_url"] = json!(source_url);
    }
    let mut runs = json!({
        "prefill_tps": metric_values(input.measurements, |value| value.prefill_tps),
        "decode_tps": metric_values(input.measurements, |value| value.decode_tps),
        "prefill_duration_ms": metric_values(input.measurements, |value| value.prefill_duration_ms),
        "decode_duration_ms": metric_values(input.measurements, |value| value.decode_duration_ms),
    });
    if input
        .measurements
        .iter()
        .all(|value| value.ttft_ms.is_some())
    {
        runs["ttft_ms"] = json!(metric_values(input.measurements, |value| value
            .ttft_ms
            .unwrap()));
    }
    if input.measurements.iter().all(|value| {
        value.wall_time_ms + 0.001 >= value.prefill_duration_ms + value.decode_duration_ms
    }) {
        runs["wall_time_ms"] = json!(metric_values(input.measurements, |value| value.wall_time_ms));
    }
    Ok(json!({
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": input.benchmark_id,
        "model": model,
        "device": {
            "name": input.args.device_name,
            "soc": input.args.soc,
        },
        "backend": backend,
        "runtime": {
            "build_command": input.args.build_command.trim(),
            "run_command": input.run_command,
        },
        "optimization": {
            "mode": input.optimization_mode,
            "methods": input.optimizations,
        },
        "workload": {
            "task": "text_generation",
            "pp": input.pp,
            "tg": input.tg,
            "context_size": input.context_size,
            "batch_size": input.batch_size,
            "concurrency": 1,
            "prompt": {
                "source": input.prompt_source,
                "sha256": input.prompt_sha256,
            },
        },
        "protocol": protocol,
        "runs": runs,
        "provenance": provenance,
    }))
}

pub(super) fn metric_values<F>(measurements: &[Measurement], selector: F) -> Vec<f64>
where
    F: Fn(&Measurement) -> f64,
{
    measurements
        .iter()
        .map(|measurement| round_metric(selector(measurement)))
        .collect()
}

pub(super) fn round_metric(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

pub(super) fn generated_benchmark_id(
    model: &str,
    backend: &str,
    timestamp: OffsetDateTime,
) -> String {
    let suffix = format!("{:08x}", timestamp.nanosecond());
    let fixed = format!("omniinfer-{}-{suffix}", timestamp.unix_timestamp());
    let available = 128usize.saturating_sub(fixed.len() + 2);
    let mut identity = format!("{}-{}", slug_component(model), slug_component(backend));
    identity.truncate(available);
    identity = identity.trim_matches('-').to_string();
    format!("{fixed}-{identity}")
}

pub(super) fn slug_component(value: &str) -> String {
    let mut slug = String::new();
    let mut separator = false;
    for character in value.chars().flat_map(char::to_lowercase) {
        if character.is_ascii_alphanumeric() {
            slug.push(character);
            separator = false;
        } else if !separator && !slug.is_empty() {
            slug.push('-');
            separator = true;
        }
    }
    slug.trim_matches('-').to_string()
}

pub(super) fn validate_benchmark_id(value: &str) -> Result<()> {
    let valid = (3..=128).contains(&value.len())
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || (index > 0 && b"._-".contains(&byte))
        });
    if !valid {
        anyhow::bail!(
            "benchmark ID must be 3-128 lowercase ASCII letters, digits, dots, underscores, or hyphens."
        );
    }
    Ok(())
}

pub(super) fn result_path(output: Option<&Path>, benchmark_id: &str) -> Result<PathBuf> {
    let path = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| paths::benchmark_results_dir().join(format!("{benchmark_id}.json")));
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

pub(super) fn write_result_atomic(destination: &Path, payload: &Value) -> Result<()> {
    if destination.exists() {
        anyhow::bail!("benchmark result already exists: {}", destination.display());
    }
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow::anyhow!("benchmark result path has no parent"))?;
    fs::create_dir_all(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if parent == paths::benchmark_results_dir() {
            fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
        }
    }
    let temporary = parent.join(format!(
        ".{}.{}-{:016x}.tmp",
        destination
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("benchmark.json"),
        std::process::id(),
        random::<u64>()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary)?;
    let write_result = (|| -> Result<()> {
        let mut raw = serde_json::to_vec_pretty(payload)?;
        raw.push(b'\n');
        file.write_all(&raw)?;
        file.sync_all()?;
        fs::rename(&temporary, destination)?;
        Ok(())
    })();
    if write_result.is_err() {
        drop(file);
        fs::remove_file(&temporary).ok();
    }
    write_result
}
