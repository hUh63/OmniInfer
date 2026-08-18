use super::*;

#[derive(Clone)]
pub(super) struct StreamHistoryContext {
    pub(super) state: GatewayState,
    pub(super) admin_id: Option<String>,
    pub(super) auth_kind: String,
    pub(super) method: String,
    pub(super) path: String,
    pub(super) model: Option<String>,
    pub(super) backend: Option<String>,
    pub(super) request: Value,
    pub(super) response_model: String,
    pub(super) started_at: Instant,
}

pub(super) fn stream_openai_chat_with_history(
    upstream_body: hyper::body::Incoming,
    builder: axum::http::response::Builder,
    context: StreamHistoryContext,
    status: StatusCode,
) -> Result<Response<Body>> {
    let (tx, rx) = mpsc::channel::<Result<HyperBytes, std::io::Error>>(16);
    tokio::spawn(async move {
        let mut body = Body::new(upstream_body);
        let mut buffered = Vec::<u8>::new();
        let mut aggregate = OpenAiStreamAggregate::new(
            &context.response_model,
            context.started_at,
            "stream_passthrough",
        );
        let mut error = None::<String>;
        while let Some(frame) = body.frame().await {
            let frame = match frame {
                Ok(frame) => frame,
                Err(frame_error) => {
                    let message = frame_error.to_string();
                    let _ = tx.send(Err(std::io::Error::other(message.clone()))).await;
                    error = Some(format!("upstream stream error: {message}"));
                    break;
                }
            };
            let Some(data) = frame.data_ref() else {
                continue;
            };
            let chunk = HyperBytes::copy_from_slice(data);
            if tx.send(Ok(chunk.clone())).await.is_err() {
                error = Some("client disconnected while streaming response".to_string());
                break;
            }
            buffered.extend_from_slice(&chunk);
            while let Some(index) = buffered.windows(2).position(|window| window == b"\n\n") {
                let event = buffered.drain(..index + 2).collect::<Vec<_>>();
                aggregate.process_sse_bytes(&event);
            }
        }
        if error.is_none() && !buffered.is_empty() {
            aggregate.process_sse_bytes(&buffered);
        }
        let payload = aggregate.finish();
        let record_error =
            error.or_else(|| (status.as_u16() >= 400).then(|| format!("HTTP {}", status.as_u16())));
        record_request_history(
            &context.state,
            RequestHistoryRecord {
                admin_id: context.admin_id,
                auth_kind: context.auth_kind,
                method: context.method,
                path: context.path,
                model: context.model,
                backend: context.backend,
                status: status.as_u16(),
                latency_ms: duration_ms(context.started_at.elapsed()),
                usage: payload.get("usage").cloned(),
                metrics: payload.get("omniinfer_metrics").cloned(),
                request: context.request,
                response: Some(payload),
                error: record_error,
            },
        );
    });
    let mut response = builder.body(Body::from_stream(ReceiverStream::new(rx)))?;
    add_cors_headers(response.headers_mut());
    Ok(response)
}

pub(super) fn should_proxy_vllm_nonstream_via_stream(
    backend_id: &str,
    stream_requested: bool,
) -> bool {
    !stream_requested
        && backend_id.starts_with("vllm")
        && env_flag_enabled("OMNIINFER_VLLM_NONSTREAM_VIA_STREAM", true)
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

pub(super) async fn proxy_openai_nonstream_via_stream(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    uri: &str,
    mut payload: Value,
    response_model: &str,
) -> Result<(Value, StatusCode)> {
    payload["stream"] = json!(true);
    ensure_stream_usage(&mut payload);
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(HyperBytes::from(serde_json::to_vec(&payload)?)))?;
    let start = Instant::now();
    let response = client.request(request).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.into_body().collect().await?.to_bytes();
        let payload = serde_json::from_slice::<Value>(&body).unwrap_or_else(
            |_| json!({"error": {"message": String::from_utf8_lossy(&body).trim().to_string()}}),
        );
        return Ok((payload, status));
    }
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !content_type.contains("text/event-stream") {
        let body = response.into_body().collect().await?.to_bytes();
        let mut payload = serde_json::from_slice::<Value>(&body).unwrap_or_else(
            |_| json!({"error": {"message": String::from_utf8_lossy(&body).trim().to_string()}}),
        );
        normalize_openai_usage(&mut payload);
        return Ok((payload, status));
    }
    let mut aggregate = OpenAiStreamAggregate::new(response_model, start, "nonstream_via_stream");
    let mut body = Body::new(response.into_body());
    let mut buffered = Vec::<u8>::new();
    while let Some(frame) = body.frame().await {
        let frame = frame?;
        let Some(data) = frame.data_ref() else {
            continue;
        };
        buffered.extend_from_slice(data);
        while let Some(index) = buffered.windows(2).position(|window| window == b"\n\n") {
            let chunk = buffered.drain(..index + 2).collect::<Vec<_>>();
            aggregate.process_sse_bytes(&chunk);
        }
    }
    if !buffered.is_empty() {
        aggregate.process_sse_bytes(&buffered);
    }
    let mut payload = aggregate.finish();
    normalize_openai_usage(&mut payload);
    Ok((payload, StatusCode::OK))
}

fn ensure_stream_usage(payload: &mut Value) {
    let object = payload
        .as_object_mut()
        .expect("normalized chat payload should be an object");
    let stream_options = object.entry("stream_options").or_insert_with(|| json!({}));
    if !stream_options.is_object() {
        *stream_options = json!({});
    }
    stream_options["include_usage"] = json!(true);
}

#[derive(Default)]
struct AggregatedToolCall {
    id: Option<String>,
    kind: Option<String>,
    name: Option<String>,
    arguments: String,
}

struct OpenAiStreamAggregate {
    metrics_mode: &'static str,
    response_model: String,
    started_at: Instant,
    first_output_at: Option<Instant>,
    id: Option<String>,
    created: Option<u64>,
    upstream_model: Option<String>,
    system_fingerprint: Option<Value>,
    role: Option<String>,
    content: String,
    content_truncated: bool,
    reasoning_content: String,
    reasoning_truncated: bool,
    tool_calls: BTreeMap<u64, AggregatedToolCall>,
    tool_arguments_truncated: bool,
    finish_reason: Option<Value>,
    usage: Option<Value>,
}

impl OpenAiStreamAggregate {
    fn new(response_model: &str, started_at: Instant, metrics_mode: &'static str) -> Self {
        Self {
            metrics_mode,
            response_model: response_model.to_string(),
            started_at,
            first_output_at: None,
            id: None,
            created: None,
            upstream_model: None,
            system_fingerprint: None,
            role: None,
            content: String::new(),
            content_truncated: false,
            reasoning_content: String::new(),
            reasoning_truncated: false,
            tool_calls: BTreeMap::new(),
            tool_arguments_truncated: false,
            finish_reason: None,
            usage: None,
        }
    }

    fn process_sse_bytes(&mut self, bytes: &[u8]) {
        for event in parse_openai_sse_events(bytes) {
            if let Ok(value) = serde_json::from_str::<Value>(&event) {
                self.process_chunk(&value);
            }
        }
    }

    fn process_chunk(&mut self, chunk: &Value) {
        if self.id.is_none() {
            self.id = chunk.get("id").and_then(Value::as_str).map(str::to_string);
        }
        if self.created.is_none() {
            self.created = chunk.get("created").and_then(Value::as_u64);
        }
        if self.upstream_model.is_none() {
            self.upstream_model = chunk
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        if self.system_fingerprint.is_none() {
            self.system_fingerprint = chunk.get("system_fingerprint").cloned();
        }
        if let Some(usage) = chunk.get("usage")
            && !usage.is_null()
        {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return;
        };
        if let Some(reason) = choice.get("finish_reason")
            && !reason.is_null()
        {
            self.finish_reason = Some(reason.clone());
        }
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if let Some(role) = delta.get("role").and_then(Value::as_str)
            && self.role.is_none()
        {
            self.role = Some(role.to_string());
        }
        if let Some(content) = delta.get("content").and_then(Value::as_str)
            && !content.is_empty()
        {
            self.mark_first_output();
            self.content_truncated |= append_limited_stream_capture(&mut self.content, content);
        }
        for key in ["reasoning_content", "reasoning"] {
            if let Some(reasoning) = delta.get(key).and_then(Value::as_str)
                && !reasoning.is_empty()
            {
                self.mark_first_output();
                self.reasoning_truncated |=
                    append_limited_stream_capture(&mut self.reasoning_content, reasoning);
            }
        }
        for tool_call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            self.mark_first_output();
            let index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let entry = self.tool_calls.entry(index).or_default();
            if let Some(id) = tool_call.get("id").and_then(Value::as_str)
                && !id.is_empty()
            {
                entry.id = Some(id.to_string());
            }
            if let Some(kind) = tool_call.get("type").and_then(Value::as_str)
                && !kind.is_empty()
            {
                entry.kind = Some(kind.to_string());
            }
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            if let Some(name) = function.get("name").and_then(Value::as_str)
                && !name.is_empty()
            {
                entry.name = Some(name.to_string());
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str)
                && !arguments.is_empty()
            {
                self.tool_arguments_truncated |=
                    append_limited_stream_capture(&mut entry.arguments, arguments);
            }
        }
    }

    fn finish(self) -> Value {
        let ended_at = Instant::now();
        let latency_ms = duration_ms(ended_at.duration_since(self.started_at));
        let ttft_ms = self
            .first_output_at
            .map(|instant| duration_ms(instant.duration_since(self.started_at)));
        let decode_ms = self
            .first_output_at
            .map(|instant| duration_ms(ended_at.duration_since(instant)));
        let mut usage = self.usage.unwrap_or_else(|| json!({}));
        normalize_openai_usage_object(&mut usage);
        let observed = observed_metrics(&usage, latency_ms, ttft_ms, decode_ms);
        let mut message = json!({
            "role": self.role.unwrap_or_else(|| "assistant".to_string()),
            "content": self.content,
        });
        if !self.reasoning_content.is_empty() {
            message["reasoning_content"] = json!(self.reasoning_content);
        }
        let tool_calls = self
            .tool_calls
            .into_iter()
            .map(|(index, tool)| {
                json!({
                    "index": index,
                    "id": tool.id.unwrap_or_else(|| format!("call_{index}")),
                    "type": tool.kind.unwrap_or_else(|| "function".to_string()),
                    "function": {
                        "name": tool.name.unwrap_or_default(),
                        "arguments": tool.arguments,
                    },
                })
            })
            .collect::<Vec<_>>();
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        let response_truncated =
            self.content_truncated || self.reasoning_truncated || self.tool_arguments_truncated;
        let mut payload = json!({
            "id": self.id.unwrap_or_else(make_chat_completion_id),
            "object": "chat.completion",
            "created": self.created.unwrap_or_else(unix_seconds),
            "model": self.upstream_model.unwrap_or(self.response_model),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason.unwrap_or(Value::String("stop".to_string())),
            }],
            "usage": usage,
            "omniinfer_metrics": {
                "mode": self.metrics_mode,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "decode_ms": decode_ms,
                "observed_prefill_tps": observed.prefill_tps,
                "observed_decode_tps": observed.decode_tps,
                "response_truncated": response_truncated,
            },
        });
        if let Some(fingerprint) = self.system_fingerprint {
            payload["system_fingerprint"] = fingerprint;
        }
        payload
    }

    fn mark_first_output(&mut self) {
        if self.first_output_at.is_none() {
            self.first_output_at = Some(Instant::now());
        }
    }
}

fn append_limited_stream_capture(target: &mut String, text: &str) -> bool {
    let current = target.chars().count();
    if current >= MAX_STREAM_HISTORY_CAPTURE_CHARS {
        return true;
    }
    let remaining = MAX_STREAM_HISTORY_CAPTURE_CHARS - current;
    let incoming = text.chars().count();
    if incoming <= remaining {
        target.push_str(text);
        return false;
    }
    target.extend(text.chars().take(remaining));
    true
}

struct ObservedMetrics {
    prefill_tps: Option<f64>,
    decode_tps: Option<f64>,
}

fn observed_metrics(
    usage: &Value,
    _latency_ms: u64,
    ttft_ms: Option<u64>,
    decode_ms: Option<u64>,
) -> ObservedMetrics {
    let prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_u64);
    let completion_tokens = usage.get("completion_tokens").and_then(Value::as_u64);
    ObservedMetrics {
        prefill_tps: tokens_per_second(prompt_tokens, ttft_ms),
        decode_tps: tokens_per_second(completion_tokens, decode_ms),
    }
}

fn tokens_per_second(tokens: Option<u64>, millis: Option<u64>) -> Option<f64> {
    let tokens = tokens?;
    let millis = millis?;
    if tokens == 0 || millis == 0 {
        return None;
    }
    Some((tokens as f64) * 1000.0 / (millis as f64))
}

pub(super) fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn make_chat_completion_id() -> String {
    format!("chatcmpl-omniinfer-{}", unix_seconds())
}

fn normalize_openai_usage_object(usage: &mut Value) {
    let Some(object) = usage.as_object_mut() else {
        return;
    };
    if object.get("total_tokens").and_then(Value::as_u64).is_some() {
        return;
    }
    let Some(prompt_tokens) = object.get("prompt_tokens").and_then(Value::as_u64) else {
        return;
    };
    let Some(completion_tokens) = object.get("completion_tokens").and_then(Value::as_u64) else {
        return;
    };
    object.insert(
        "total_tokens".to_string(),
        json!(prompt_tokens.saturating_add(completion_tokens)),
    );
}

pub(super) fn apply_proxy_model(payload: &mut Value, proxy_model: Option<&str>) {
    if let Some(proxy_model) = proxy_model.filter(|value| !value.trim().is_empty()) {
        payload["model"] = json!(proxy_model);
    }
}
