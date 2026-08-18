use super::*;

pub(super) fn iter_local_models(root: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    visit_models(root, &mut result);
    result.sort();
    result
}

pub(super) fn visit_models(root: &Path, result: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit_models(&path, result);
        } else if path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
            && !path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                .contains("mmproj")
        {
            result.push(path);
        }
    }
}

pub(super) fn task_matches_model(task: &str, model_info: &Value) -> bool {
    let normalized = task.trim().to_ascii_lowercase();
    let model_name = json_str(model_info, "model")
        .unwrap_or("")
        .to_ascii_lowercase();
    let caps = model_info
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    match normalized.as_str() {
        "any" => true,
        "chat" | "general" => caps.contains(&"chat"),
        "vision" | "multimodal" => caps.contains(&"vision"),
        "action" | "robotics" | "vla" => caps.contains(&"action"),
        "embedding" | "embeddings" => caps.contains(&"embedding"),
        "coding" => ["coder", "code", "deepseek", "qwen"]
            .iter()
            .any(|token| model_name.contains(token)),
        _ => true,
    }
}

pub(super) fn recommendation_score(candidate: &Value, model_info: &Value) -> f64 {
    let fit_score = match json_str(candidate, "fit").unwrap_or("unknown") {
        "good" => 100.0,
        "marginal" => 65.0,
        "too_tight" => 10.0,
        _ => 20.0,
    };
    let installed_bonus = if json_bool(candidate, "installed").unwrap_or(false) {
        10.0
    } else {
        0.0
    };
    let priority_penalty = json_u64(candidate, "priority").unwrap_or(0) as f64 * 2.0;
    let size_bonus = model_info
        .get("size_gib")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .min(30.0)
        / 3.0;
    round_gib(fit_score + installed_bonus + size_bonus - priority_penalty)
}

pub(super) fn shell_quote(value: &str) -> String {
    if value.chars().all(|ch| {
        ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '/' | ':' | '=' | '+' | '-')
    }) {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\"'\"'"))
    }
}

pub(super) fn round_gib(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

pub(super) fn format_gib(value: f64) -> String {
    let rounded = round_gib(value);
    if (rounded.fract()).abs() < f64::EPSILON {
        format!("{rounded:.1}")
    } else {
        format!("{rounded:.2}")
    }
}
