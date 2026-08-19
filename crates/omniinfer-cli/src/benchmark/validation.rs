use super::*;

pub(super) fn validate_metadata(args: &BenchRunArgs) -> Result<()> {
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
    if let Some(notes) = args.notes.as_deref() {
        validate_text("--notes", notes, 2048)?;
    }
    if let Some(value) = protocol_notes(args)
        && value.chars().count() > 2048
    {
        anyhow::bail!(
            "generated protocol notes exceed 2048 characters after the --ignore-eos marker is appended. Shorten --notes."
        );
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

pub(super) fn validate_text(label: &str, value: &str, max_chars: usize) -> Result<()> {
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

pub(super) fn validate_https_url(label: &str, value: &str) -> Result<()> {
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

pub(super) fn validated_command<'a>(label: &str, command: &'a str) -> Result<&'a str> {
    validate_text(label, command, 16_384)?;
    if contains_unredacted_secret(command) {
        anyhow::bail!("{label} contains an unredacted credential value.");
    }
    Ok(command.trim())
}

pub(super) fn contains_unredacted_secret(command: &str) -> bool {
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

pub(super) fn is_secret_name(value: &str) -> bool {
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

pub(super) fn is_redacted_reference(value: &str) -> bool {
    let value = value.trim_matches(['\'', '"']);
    value == "<redacted>" || value.starts_with('$') || value.starts_with('%')
}

pub(super) fn command_array(state: &Value, key: &str) -> Result<Vec<String>> {
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

pub(super) fn redact_command_args(args: &[String]) -> Vec<String> {
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

pub(super) fn command_text(args: &[String]) -> String {
    args.iter()
        .map(|argument| quote_command_argument(argument))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(not(windows))]
pub(super) fn quote_command_argument(argument: &str) -> String {
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
pub(super) fn quote_command_argument(argument: &str) -> String {
    if !argument.is_empty()
        && !argument
            .chars()
            .any(|character| character.is_whitespace() || character == '"')
    {
        return argument.to_string();
    }
    format!("\"{}\"", argument.replace('"', "\\\""))
}

pub(super) fn detect_optimizations(backend: &str, launch_args: &[String]) -> BTreeSet<String> {
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

pub(super) fn resolve_optimization_declaration(
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

pub(super) fn valid_optimization_slug(value: &str) -> bool {
    !matches!(value, "none" | "baseline" | "default")
        && (1..=64).contains(&value.len())
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || (index > 0 && b"._-".contains(&byte))
        })
}

pub(super) fn valid_catalog_id(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric() || (index > 0 && b"._-".contains(&byte))
        })
}

pub(super) fn infer_positive_flag(args: &[String], flags: &[&str]) -> Option<u32> {
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
