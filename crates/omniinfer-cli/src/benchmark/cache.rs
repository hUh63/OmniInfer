use super::*;

pub(super) fn validate_cache_isolation(
    backend: &str,
    launch_args: &[String],
    state: &Value,
) -> Result<()> {
    if state.get("mmproj").is_some_and(|value| !value.is_null()) {
        anyhow::bail!(
            "The standard text benchmark cannot use a loaded mmproj. Reload the text model without --mmproj."
        );
    }
    if !(backend.starts_with("llama.cpp-") || backend.contains("turboquant")) {
        anyhow::bail!(
            "Backend {backend} cannot currently prove per-run cache erasure. OmniInfer bench fails closed instead of labeling a cached run as cold."
        );
    }
    let cache_ram = last_option_value(launch_args, &["-cram", "--cache-ram"]);
    let similarity = last_option_value(launch_args, &["--slot-prompt-similarity"]);
    let slot_path = last_option_value(launch_args, &["--slot-save-path"]);
    let cache_prompt = last_toggle(launch_args, "--cache-prompt", "--no-cache-prompt");
    let cache_idle = last_toggle(launch_args, "--cache-idle-slots", "--no-cache-idle-slots");
    if cache_ram.as_deref() != Some("0")
        || similarity
            .as_deref()
            .and_then(|value| value.parse::<f64>().ok())
            != Some(0.0)
        || cache_prompt != Some(false)
        || cache_idle != Some(false)
        || slot_path.as_deref().is_none_or(str::is_empty)
    {
        anyhow::bail!(
            "The loaded llama.cpp runtime is not benchmark-isolated. Reload it with `-- --cache-ram 0 --no-cache-idle-slots --no-cache-prompt --slot-prompt-similarity 0 --slot-save-path <empty-directory>`."
        );
    }
    Ok(())
}

fn last_option_value(args: &[String], names: &[&str]) -> Option<String> {
    let mut result = None;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        for name in names {
            if argument == name {
                result = args.get(index + 1).cloned();
                index += 1;
                break;
            }
            if let Some(value) = argument.strip_prefix(&format!("{name}=")) {
                result = Some(value.to_string());
                break;
            }
        }
        index += 1;
    }
    result
}

fn last_toggle(args: &[String], enabled: &str, disabled: &str) -> Option<bool> {
    args.iter().fold(None, |value, argument| {
        if argument == enabled {
            Some(true)
        } else if argument == disabled {
            Some(false)
        } else {
            value
        }
    })
}
