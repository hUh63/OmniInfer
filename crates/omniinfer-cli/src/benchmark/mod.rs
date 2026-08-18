use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use omniinfer_core::{config, paths};
use rand::random;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use url::Url;

use crate::{
    BenchRunArgs, get_local_json, json_bool, json_str, json_u64, post_local_json_for_config,
};

const BENCHMARK_SCHEMA_VERSION: &str = "1.2.0";
const DEFAULT_PROMPT: &str = "Write a detailed but concise explanation of why local language-model inference speed varies across hardware and runtimes.";
const MODEL_FORMATS: &[&str] = &[
    "GGUF",
    "MLX",
    "Safetensors",
    "MNN",
    "TFLite",
    "LiteRT",
    "LITERTLM",
    "ONNX",
    "ExecuTorch",
    "Other",
];

#[derive(Debug, Clone)]
struct Measurement {
    prompt_tokens: u64,
    completion_tokens: u64,
    prefill_tps: f64,
    decode_tps: f64,
    prefill_duration_ms: f64,
    decode_duration_ms: f64,
    ttft_ms: Option<f64>,
    wall_time_ms: f64,
}

pub(crate) fn run(args: &BenchRunArgs) -> Result<()> {
    validate_metadata(args)?;
    let config = config::load_app_config().unwrap_or_default();
    let state = get_local_json("/omni/state", Duration::from_secs(10))?;
    if !json_bool(&state, "backend_ready").unwrap_or(false) {
        anyhow::bail!(
            "No benchmarkable runtime is ready. Load a model first with `omniinfer load -m <model>`."
        );
    }
    json_str(&state, "model")
        .ok_or_else(|| anyhow::anyhow!("OmniInfer state does not identify the loaded model."))?;
    let loaded_backend = json_str(&state, "backend")
        .ok_or_else(|| anyhow::anyhow!("OmniInfer state does not identify the loaded backend."))?;
    if !valid_catalog_id(loaded_backend) {
        anyhow::bail!(
            "Loaded backend ID {loaded_backend:?} cannot be represented by the benchmark schema."
        );
    }
    if let Some(expected) = args.backend_id.as_deref()
        && expected != loaded_backend
    {
        anyhow::bail!("Loaded backend is {loaded_backend}, but --backend-id requested {expected}.");
    }

    let launch_args = command_array(&state, "launch_command")?;
    let detected = detect_optimizations(loaded_backend, &launch_args);
    let (optimization_mode, optimizations) = resolve_optimization_declaration(args, &detected)?;
    let run_command = match args.run_command.as_deref() {
        Some(command) => validated_command("--run-command", command)?.to_string(),
        None if launch_args.is_empty() => anyhow::bail!(
            "The loaded runtime does not expose a launch command. Pass --run-command with the effective runtime command."
        ),
        None => validated_command(
            "captured runtime command",
            &command_text(&redact_command_args(&launch_args)),
        )?
        .to_string(),
    };

    let context_size = args
        .context_size
        .or_else(|| json_u64(&state, "ctx_size").and_then(|value| u32::try_from(value).ok()))
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Context size is unavailable from the loaded runtime. Pass --context-size explicitly."
            )
        })?;
    let batch_size = args
        .batch_size
        .or_else(|| infer_positive_flag(&launch_args, &["-b", "--batch-size", "--batch_size"]))
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Batch size is unavailable from the launch command. Pass --batch-size explicitly."
            )
        })?;
    let (prompt, prompt_source) = read_prompt(args)?;
    let prompt_sha256 = format!("{:x}", Sha256::digest(prompt.as_bytes()));
    let request = json!({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": false,
        "think": false,
    });
    let timeout = Duration::from_secs(u64::from(args.timeout_seconds));

    for index in 0..args.warmup_runs {
        post_local_json_for_config("/v1/chat/completions", &request, timeout, &config)
            .with_context(|| format!("warmup run {} failed", index + 1))?;
    }

    let started_at = OffsetDateTime::now_utc();
    let mut measurements = Vec::with_capacity(usize::from(args.runs));
    for index in 0..args.runs {
        let started = Instant::now();
        let response =
            post_local_json_for_config("/v1/chat/completions", &request, timeout, &config)
                .with_context(|| format!("measured run {} failed", index + 1))?;
        let measurement = extract_measurement(&response, started.elapsed())
            .with_context(|| format!("measured run {} returned incomplete metrics", index + 1))?;
        let progress = format!(
            "Run {}/{}: pp={} tg={} prefill={:.3} tok/s decode={:.3} tok/s",
            index + 1,
            args.runs,
            measurement.prompt_tokens,
            measurement.completion_tokens,
            measurement.prefill_tps,
            measurement.decode_tps,
        );
        if args.json {
            eprintln!("{progress}");
        } else {
            println!("{progress}");
        }
        measurements.push(measurement);
    }

    let pp = consistent_token_count(&measurements, true)?;
    let tg = consistent_token_count(&measurements, false)?;
    if pp > 1_048_576 || tg > 1_048_576 {
        anyhow::bail!("Measured PP/TG exceeds the submission contract limit of 1,048,576 tokens.");
    }
    if context_size > 4_194_304 {
        anyhow::bail!("Context size exceeds the submission contract limit of 4,194,304.");
    }
    if batch_size > 1_048_576 {
        anyhow::bail!("Batch size exceeds the submission contract limit of 1,048,576.");
    }
    if u64::from(context_size) < pp.saturating_add(tg) {
        anyhow::bail!(
            "Measured pp + tg is {}, which exceeds context size {context_size}.",
            pp.saturating_add(tg)
        );
    }
    let benchmark_id = args.benchmark_id.clone().unwrap_or_else(|| {
        generated_benchmark_id(&args.catalog_model_id, loaded_backend, started_at)
    });
    validate_benchmark_id(&benchmark_id)?;
    let payload = build_submission(BuildSubmission {
        args,
        benchmark_id: &benchmark_id,
        loaded_backend,
        run_command: &run_command,
        optimization_mode,
        optimizations: &optimizations,
        context_size,
        batch_size,
        prompt_source: &prompt_source,
        prompt_sha256: &prompt_sha256,
        started_at,
        measurements: &measurements,
        pp,
        tg,
    })?;
    let destination = result_path(args.output.as_deref(), &benchmark_id)?;
    write_result_atomic(&destination, &payload)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        eprintln!("Benchmark saved: {}", destination.display());
        eprintln!("Schema: {BENCHMARK_SCHEMA_VERSION}");
    } else {
        println!("Benchmark saved: {}", destination.display());
        println!("Schema: {BENCHMARK_SCHEMA_VERSION}");
    }
    Ok(())
}

pub(crate) fn list(json_output: bool) -> Result<()> {
    let directory = paths::benchmark_results_dir();
    let mut rows = Vec::new();
    if directory.is_dir() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json")
                || path.is_symlink()
            {
                continue;
            }
            let Ok(raw) = fs::read(&path) else {
                continue;
            };
            let Ok(payload) = serde_json::from_slice::<Value>(&raw) else {
                continue;
            };
            if json_str(&payload, "schema_version") != Some(BENCHMARK_SCHEMA_VERSION) {
                continue;
            }
            rows.push(json!({
                "benchmark_id": json_str(&payload, "benchmark_id"),
                "model_id": payload.pointer("/model/catalog_model_id").and_then(Value::as_str),
                "backend_id": payload.pointer("/backend/catalog_backend_id").and_then(Value::as_str),
                "started_at": payload.pointer("/protocol/started_at").and_then(Value::as_str),
                "path": path.display().to_string(),
            }));
        }
    }
    rows.sort_by(|left, right| {
        left["started_at"]
            .as_str()
            .cmp(&right["started_at"].as_str())
            .reverse()
    });
    if json_output {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if rows.is_empty() {
        println!("No archived benchmark results in {}", directory.display());
        return Ok(());
    }
    println!("Archived benchmark results:");
    for row in rows {
        println!(
            "  {}  model={}  backend={}  {}",
            json_str(&row, "benchmark_id").unwrap_or("-"),
            json_str(&row, "model_id").unwrap_or("-"),
            json_str(&row, "backend_id").unwrap_or("-"),
            json_str(&row, "path").unwrap_or("-"),
        );
    }
    Ok(())
}

fn validate_metadata(args: &BenchRunArgs) -> Result<()> {
    for (label, value, max) in [
        ("--catalog-model-id", args.catalog_model_id.as_str(), 128),
        ("--quantization", args.quantization.as_str(), 256),
        ("--device-name", args.device_name.as_str(), 256),
        ("--soc", args.soc.as_str(), 256),
        ("--backend-version", args.backend_version.as_str(), 256),
        ("--submitter-name", args.submitter_name.as_str(), 256),
    ] {
        validate_text(label, value, max)?;
    }
    if !MODEL_FORMATS.contains(&args.model_format.as_str()) {
        anyhow::bail!("--format must be one of: {}", MODEL_FORMATS.join(", "));
    }
    validate_https_url("--model-url", &args.model_url)?;
    if args.model_url.len() > 2048 {
        anyhow::bail!("--model-url exceeds 2048 characters.");
    }
    if let Some(value) = args.source_url.as_deref() {
        validate_https_url("--source-url", value)?;
        if value.len() > 2048 {
            anyhow::bail!("--source-url exceeds 2048 characters.");
        }
    }
    for (label, value) in [
        ("--model-name", args.model_name.as_deref()),
        ("--backend-id", args.backend_id.as_deref()),
        ("--backend-name", args.backend_name.as_deref()),
        ("--organization", args.organization.as_deref()),
    ] {
        if let Some(value) = value {
            validate_text(label, value, 256)?;
        }
    }
    if let Some(value) = args.notes.as_deref() {
        validate_text("--notes", value, 2048)?;
    }
    validated_command("--build-command", &args.build_command)?;
    if !valid_catalog_id(&args.catalog_model_id) {
        anyhow::bail!("--catalog-model-id must be a 1-128 character catalog slug.");
    }
    if let Some(value) = args.backend_id.as_deref()
        && !valid_catalog_id(value)
    {
        anyhow::bail!("--backend-id must be a 1-128 character catalog slug.");
    }
    if args.baseline && !args.optimizations.is_empty() {
        anyhow::bail!("--baseline conflicts with --optimization.");
    }
    if !args.baseline && args.optimizations.is_empty() {
        anyhow::bail!(
            "Explicit optimization declaration is required. Pass --baseline or at least one --optimization <slug>."
        );
    }
    Ok(())
}

fn validate_text(label: &str, value: &str, max_chars: usize) -> Result<()> {
    let value = value.trim();
    if value.is_empty() {
        anyhow::bail!("{label} must not be empty.");
    }
    if value.chars().count() > max_chars {
        anyhow::bail!("{label} exceeds {max_chars} characters.");
    }
    if value
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
    {
        anyhow::bail!("{label} contains a control character.");
    }
    Ok(())
}

fn validate_https_url(label: &str, value: &str) -> Result<()> {
    let parsed = Url::parse(value).with_context(|| format!("{label} is not a valid URL"))?;
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
    {
        anyhow::bail!("{label} must be a public HTTPS URL without credentials or a fragment.");
    }
    Ok(())
}

fn validated_command<'a>(label: &str, command: &'a str) -> Result<&'a str> {
    validate_text(label, command, 16_384)?;
    if contains_unredacted_secret(command) {
        anyhow::bail!("{label} contains an unredacted credential value.");
    }
    Ok(command.trim())
}

fn contains_unredacted_secret(command: &str) -> bool {
    let tokens = command.split_whitespace().collect::<Vec<_>>();
    for (index, token) in tokens.iter().enumerate() {
        if let Some((key, value)) = token.split_once('=') {
            if is_secret_name(key) && !is_redacted_reference(value) {
                return true;
            }
        } else if is_secret_name(token)
            && tokens
                .get(index + 1)
                .is_some_and(|value| !is_redacted_reference(value))
        {
            return true;
        }
    }
    false
}

fn is_secret_name(value: &str) -> bool {
    let normalized = value
        .trim_start_matches('-')
        .to_ascii_lowercase()
        .replace('-', "_");
    normalized.contains("api_key")
        || normalized == "token"
        || normalized.ends_with("_token")
        || normalized.contains("password")
        || normalized.contains("secret")
}

fn is_redacted_reference(value: &str) -> bool {
    let value = value.trim_matches(['\'', '"']);
    value == "<redacted>" || value.starts_with('$') || value.starts_with('%')
}

fn command_array(state: &Value, key: &str) -> Result<Vec<String>> {
    let Some(values) = state.get(key).and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .map(|value| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                anyhow::anyhow!("OmniInfer state field {key} is not a string array.")
            })
        })
        .collect()
}

fn redact_command_args(args: &[String]) -> Vec<String> {
    let mut redacted = Vec::with_capacity(args.len());
    let mut redact_next = false;
    for argument in args {
        if redact_next {
            redacted.push("<redacted>".to_string());
            redact_next = false;
            continue;
        }
        if let Some((key, _value)) = argument.split_once('=') {
            if is_secret_name(key) {
                redacted.push(format!("{key}=<redacted>"));
            } else {
                redacted.push(argument.clone());
            }
            continue;
        }
        redact_next = is_secret_name(argument);
        redacted.push(argument.clone());
    }
    redacted
}

fn command_text(args: &[String]) -> String {
    args.iter()
        .map(|argument| quote_command_argument(argument))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(not(windows))]
fn quote_command_argument(argument: &str) -> String {
    if !argument.is_empty()
        && argument
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_@%+=:,./-".contains(character))
    {
        return argument.to_string();
    }
    format!("'{}'", argument.replace('\'', "'\"'\"'"))
}

#[cfg(windows)]
fn quote_command_argument(argument: &str) -> String {
    if !argument.is_empty()
        && !argument
            .chars()
            .any(|character| character.is_whitespace() || character == '"')
    {
        return argument.to_string();
    }
    format!("\"{}\"", argument.replace('"', "\\\""))
}

fn detect_optimizations(backend: &str, launch_args: &[String]) -> BTreeSet<String> {
    let mut methods = BTreeSet::new();
    let searchable = std::iter::once(backend)
        .chain(launch_args.iter().map(String::as_str))
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    if searchable.iter().any(|value| value.contains("dflash")) {
        methods.insert("dflash".to_string());
    }
    if searchable.iter().any(|value| value.contains("turboquant")) {
        methods.insert("turboquant".to_string());
    }
    methods
}

fn resolve_optimization_declaration(
    args: &BenchRunArgs,
    detected: &BTreeSet<String>,
) -> Result<(&'static str, Vec<String>)> {
    let declared = args
        .optimizations
        .iter()
        .map(|method| method.trim().to_string())
        .collect::<BTreeSet<_>>();
    if declared.len() != args.optimizations.len() {
        anyhow::bail!("--optimization values must be unique.");
    }
    for method in &declared {
        if !valid_optimization_slug(method) {
            anyhow::bail!("invalid --optimization slug: {method}");
        }
    }
    if args.baseline {
        if !detected.is_empty() {
            anyhow::bail!(
                "Runtime state indicates active optimization(s): {}. Do not declare --baseline.",
                detected.iter().cloned().collect::<Vec<_>>().join(", ")
            );
        }
        return Ok(("baseline", Vec::new()));
    }
    for method in detected {
        if !declared
            .iter()
            .any(|declared| declared == method || declared.starts_with(&format!("{method}-")))
        {
            anyhow::bail!(
                "Runtime state indicates optimization {method}. Rerun with --optimization {method} so the declaration is explicit."
            );
        }
    }
    Ok(("optimized", declared.into_iter().collect()))
}

fn valid_optimization_slug(value: &str) -> bool {
    !matches!(value, "none" | "baseline" | "default")
        && (1..=64).contains(&value.len())
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || (index > 0 && b"._-".contains(&byte))
        })
}

fn valid_catalog_id(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric() || (index > 0 && b"._-".contains(&byte))
        })
}

fn infer_positive_flag(args: &[String], flags: &[&str]) -> Option<u32> {
    for (index, argument) in args.iter().enumerate() {
        for flag in flags {
            if argument == flag
                && let Some(value) = args.get(index + 1).and_then(|value| value.parse().ok())
            {
                return Some(value);
            }
            if let Some(value) = argument.strip_prefix(&format!("{flag}="))
                && let Ok(value) = value.parse()
            {
                return Some(value);
            }
        }
    }
    None
}

fn read_prompt(args: &BenchRunArgs) -> Result<(String, String)> {
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

fn extract_measurement(response: &Value, elapsed: Duration) -> Result<Measurement> {
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

fn positive_number(object: Option<&Map<String, Value>>, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| object?.get(*key)?.as_f64())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn consistent_token_count(measurements: &[Measurement], prompt: bool) -> Result<u64> {
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

struct BuildSubmission<'a> {
    args: &'a BenchRunArgs,
    benchmark_id: &'a str,
    loaded_backend: &'a str,
    run_command: &'a str,
    optimization_mode: &'a str,
    optimizations: &'a [String],
    context_size: u32,
    batch_size: u32,
    prompt_source: &'a str,
    prompt_sha256: &'a str,
    started_at: OffsetDateTime,
    measurements: &'a [Measurement],
    pp: u64,
    tg: u64,
}

fn build_submission(input: BuildSubmission<'_>) -> Result<Value> {
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
    if let Some(notes) = input.args.notes.as_deref() {
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

fn metric_values<F>(measurements: &[Measurement], selector: F) -> Vec<f64>
where
    F: Fn(&Measurement) -> f64,
{
    measurements
        .iter()
        .map(|measurement| round_metric(selector(measurement)))
        .collect()
}

fn round_metric(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn generated_benchmark_id(model: &str, backend: &str, timestamp: OffsetDateTime) -> String {
    let suffix = format!("{:08x}", timestamp.nanosecond());
    let fixed = format!("omniinfer-{}-{suffix}", timestamp.unix_timestamp());
    let available = 128usize.saturating_sub(fixed.len() + 2);
    let mut identity = format!("{}-{}", slug_component(model), slug_component(backend));
    identity.truncate(available);
    identity = identity.trim_matches('-').to_string();
    format!("{fixed}-{identity}")
}

fn slug_component(value: &str) -> String {
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

fn validate_benchmark_id(value: &str) -> Result<()> {
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

fn result_path(output: Option<&Path>, benchmark_id: &str) -> Result<PathBuf> {
    let path = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| paths::benchmark_results_dir().join(format!("{benchmark_id}.json")));
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn write_result_atomic(destination: &Path, payload: &Value) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_sensitive_runtime_arguments() {
        let args = vec![
            "runtime".to_string(),
            "--api-key".to_string(),
            "secret-value".to_string(),
            "TOKEN=another-secret".to_string(),
            "--port".to_string(),
            "9000".to_string(),
        ];
        assert_eq!(
            redact_command_args(&args),
            vec![
                "runtime",
                "--api-key",
                "<redacted>",
                "TOKEN=<redacted>",
                "--port",
                "9000",
            ]
        );
    }

    #[test]
    fn detects_known_optimization_markers() {
        let detected = detect_optimizations(
            "turboquant-mac",
            &["runtime".to_string(), "--dflash-mode".to_string()],
        );
        assert_eq!(
            detected.into_iter().collect::<Vec<_>>(),
            vec!["dflash", "turboquant"]
        );
    }

    #[test]
    fn extracts_backend_and_observed_measurements() {
        let backend = json!({
            "usage": {"prompt_tokens": 64, "completion_tokens": 16},
            "timings": {
                "prompt_ms": 400.0,
                "predicted_ms": 800.0
            }
        });
        let value = extract_measurement(&backend, Duration::from_millis(1300)).unwrap();
        assert_eq!(value.prefill_tps, 160.0);
        assert_eq!(value.decode_tps, 20.0);

        let observed = json!({
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "omniinfer_metrics": {
                "ttft_ms": 100,
                "decode_ms": 500,
                "observed_prefill_tps": 200.0,
                "observed_decode_tps": 20.0
            }
        });
        let value = extract_measurement(&observed, Duration::from_millis(650)).unwrap();
        assert_eq!(value.prefill_tps, 200.0);
        assert_eq!(value.decode_tps, 20.0);
        assert_eq!(value.ttft_ms, Some(100.0));

        let upstream_rates = json!({
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "timings": {
                "prompt_per_second": 400.0,
                "predicted_per_second": 40.0
            },
            "omniinfer_metrics": {
                "ttft_ms": 100,
                "decode_ms": 500,
                "observed_prefill_tps": 200.0,
                "observed_decode_tps": 20.0
            }
        });
        let value = extract_measurement(&upstream_rates, Duration::from_millis(650)).unwrap();
        assert_eq!(value.prefill_tps, 400.0);
        assert_eq!(value.decode_tps, 40.0);
    }

    #[test]
    fn infers_common_batch_flags() {
        assert_eq!(
            infer_positive_flag(&["runtime".into(), "-b".into(), "256".into()], &["-b"]),
            Some(256)
        );
        assert_eq!(
            infer_positive_flag(
                &["runtime".into(), "--batch-size=64".into()],
                &["--batch-size"]
            ),
            Some(64)
        );
    }
}
