use super::*;

#[tokio::test]
async fn rust_gateway_rejects_embedded_model_loads() {
    let Some(backend_id) = embedded_test_backend_id() else {
        return;
    };
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-embedded");
    std::fs::create_dir_all(&temp).unwrap();
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;

    let load_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(json!({
                "backend": backend_id,
                "model": "embedded-demo",
                "ctx_size": 512
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(load_response.status().as_u16(), 400);
    let load_body: Value = load_response.into_body().read_json().unwrap();
    assert!(
        load_body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("embedded backend")
    );
    assert!(
        load_body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Python control-plane fallback has been removed")
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn anthropic_stream_response_emits_before_backend_finishes() {
    let (tx, rx) = mpsc::channel::<Result<HyperBytes, std::io::Error>>(4);
    let backend_body = Body::from_stream(ReceiverStream::new(rx));
    let response = anthropic_stream_response(backend_body, "claude-compatible".to_string());
    let mut body = response.into_body();

    tx.send(Ok(HyperBytes::from_static(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"fake\"}}]}\n\n",
    )))
    .await
    .unwrap();

    let first = tokio::time::timeout(Duration::from_millis(300), body.frame())
        .await
        .expect("first Anthropic stream frame should arrive before backend completes")
        .expect("body frame")
        .expect("body frame ok")
        .into_data()
        .expect("data frame");
    let first_text = String::from_utf8(first.to_vec()).unwrap();
    assert!(first_text.contains("event: message_start"));

    tx.send(Ok(HyperBytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\" backend\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
        )))
        .await
        .unwrap();
    tx.send(Ok(HyperBytes::from_static(b"data: [DONE]\n\n")))
        .await
        .unwrap();
    drop(tx);
    let rest = body.collect().await.unwrap().to_bytes();
    let rest_text = String::from_utf8(rest.to_vec()).unwrap();
    assert!(rest_text.contains("\"text\":\"fake\""));
    assert!(rest_text.contains("\"text\":\" backend\""));
    assert!(rest_text.contains("event: message_stop"));
}

#[tokio::test]
async fn rust_gateway_discovers_model_directory_artifacts() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-artifacts");
    let model_dir = temp.join("models").join("vision-model");
    let nested = model_dir.join("nested");
    std::fs::create_dir_all(&nested).unwrap();
    let model = nested.join("model.gguf");
    let mmproj = model_dir.join("mmproj-F16.gguf");
    std::fs::write(&model, "").unwrap();
    std::fs::write(&mmproj, "").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;

    let load_response = tokio::task::spawn_blocking({
        let model_dir = model_dir.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(json!({
                    "backend": backend_id,
                    "model": model_dir.display().to_string(),
                    "ctx_size": 512
                }))
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(load_response.status().as_u16(), 200);
    let load_body: Value = load_response.into_body().read_json().unwrap();
    assert_eq!(
        load_body["selected_model"].as_str().unwrap(),
        model.display().to_string()
    );
    assert_eq!(
        load_body["selected_mmproj"].as_str().unwrap(),
        mmproj.display().to_string()
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn public_models_requires_admin_key_for_remote_clients() {
    let temp = temp_root("rust-gateway-public-models-auth");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "qwen3.5-4b-q4_k_m");
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let denied = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/public-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .call()
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(denied.to_string().contains("401"));

    let allowed = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/public-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(allowed.status().as_u16(), 200);
    let body: Value = allowed.into_body().read_json().unwrap();
    assert_eq!(body["data"][0]["id"], "qwen3.5-4b-q4_k_m");

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn remote_public_model_select_resolves_model_id() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-public-model-select");
    let root = temp.join("public_models");
    let model_path = write_public_model_manifest(&root, "qwen3.5-4b-q4_k_m");
    install_fake_llama_server(&temp, external_test_backend_id());
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin")
            .send_json(json!({"model": "qwen3.5-4b-q4_k_m"}))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let body: Value = response.into_body().read_json().unwrap();
    assert_eq!(body["selected_model"], model_path.display().to_string());
    assert_eq!(body["selected_backend"], external_test_backend_id());
    assert_eq!(body["selected_ctx_size"], 512);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn vllm_public_model_rewrites_to_served_model_name() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-vllm-public-model");
    let root = temp.join("public_models");
    write_vllm_public_model_manifest(&root, "gelab-zero-4b-preview");
    install_fake_vllm_server(&temp);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());
    let _stream_guard = EnvGuard::set("OMNIINFER_VLLM_NONSTREAM_VIA_STREAM", "0".to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let load = remote_admin_post(
        port,
        "/omni/model/select",
        "admin",
        json!({"model": "gelab-zero-4b-preview"}),
    )
    .await;
    assert_eq!(load["selected_backend"], "vllm-linux-cuda");
    assert_eq!(load["model"], "gelab-zero-4b-preview");

    let chat = remote_chat(port, "inference", "gelab-zero-4b-preview").await;
    assert_eq!(chat["model_echo"], "local");
    assert_eq!(chat["usage"]["prompt_tokens"], 3);
    assert_eq!(chat["usage"]["completion_tokens"], 2);
    assert_eq!(chat["usage"]["total_tokens"], 5);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn vllm_nonstream_can_be_aggregated_from_stream() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-vllm-nonstream-via-stream");
    let root = temp.join("public_models");
    write_vllm_public_model_manifest(&root, "gelab-zero-4b-preview");
    install_fake_vllm_server(&temp);
    let _state_guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let load = remote_admin_post(
        port,
        "/omni/model/select",
        "admin",
        json!({"model": "gelab-zero-4b-preview"}),
    )
    .await;
    assert_eq!(load["selected_backend"], "vllm-linux-cuda");

    let chat = remote_chat(port, "inference", "gelab-zero-4b-preview").await;
    assert_eq!(chat["object"], "chat.completion");
    assert_eq!(chat["model"], "local");
    assert_eq!(chat["choices"][0]["message"]["content"], "fake backend");
    assert_eq!(chat["choices"][0]["finish_reason"], "stop");
    assert_eq!(chat["usage"]["prompt_tokens"], 3);
    assert_eq!(chat["usage"]["completion_tokens"], 2);
    assert_eq!(chat["usage"]["total_tokens"], 5);
    assert_eq!(chat["omniinfer_metrics"]["mode"], "nonstream_via_stream");
    assert!(chat["omniinfer_metrics"]["latency_ms"].as_u64().is_some());
    assert!(
        chat["omniinfer_metrics"]
            .get("observed_decode_tps")
            .is_some()
    );

    let denied = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/request-history"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .call()
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(denied.to_string().contains("401") || denied.to_string().contains("403"));

    let history = wait_for_history(port, "admin", "gelab-zero-4b-preview").await;
    assert_eq!(history["data"][0]["model"], "gelab-zero-4b-preview");
    assert_eq!(history["data"][0]["backend"], "vllm-linux-cuda");
    assert_eq!(history["data"][0]["status"], 200);
    assert_eq!(
        history["data"][0]["response"]["choices"][0]["message"]["content"],
        "fake backend"
    );
    assert_eq!(
        history["data"][0]["response"]["omniinfer_metrics"]["mode"],
        "nonstream_via_stream"
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}
