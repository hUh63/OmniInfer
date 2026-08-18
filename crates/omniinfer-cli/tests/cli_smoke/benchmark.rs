use super::support::*;

#[test]
fn bench_archives_submission_compatible_json() {
    let state = r#"{
        "backend_ready": true,
        "model": "/models/qwen.gguf",
        "backend": "llama.cpp-linux-cuda",
        "ctx_size": 128,
        "launch_command": [
            "llama-server", "-m", "/models/qwen.gguf", "-b", "64",
            "--api-key", "runtime-secret"
        ]
    }"#;
    let measurement = r#"{
        "usage": {"prompt_tokens": 64, "completion_tokens": 16},
        "timings": {"prompt_ms": 400.0, "predicted_ms": 800.0}
    }"#;
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(state),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
    ]);
    let root = temp_repo_root("bench-run");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");

    let benchmark_id = "contract-test-omniinfer-bench";
    let mut command = Command::cargo_bin("omniinfer").expect("binary exists");
    let assert = command
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args([
            "bench",
            "run",
            "--benchmark-id",
            benchmark_id,
            "--catalog-model-id",
            "qwen3-5-2b",
            "--format",
            "GGUF",
            "--quantization",
            "Q4_0",
            "--model-url",
            "https://example.com/qwen.gguf",
            "--device-name",
            "Test GPU",
            "--soc",
            "test-gpu",
            "--backend-version",
            "test-runtime-1",
            "--build-command",
            "bash scripts/build-test-runtime.sh",
            "--baseline",
            "--runs",
            "3",
            "--warmup-runs",
            "0",
            "--submitter-name",
            "OmniInfer Test",
            "--json",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Run 3/3"))
        .stderr(predicate::str::contains("Schema: 1.2.0"));
    let printed_payload: serde_json::Value = serde_json::from_slice(&assert.get_output().stdout)
        .expect("--json stdout is one JSON value");
    assert_eq!(printed_payload["benchmark_id"], benchmark_id);

    assert!(gateway.request().starts_with("GET /health HTTP/1.1"));
    assert!(gateway.request().starts_with("GET /omni/state HTTP/1.1"));
    for _ in 0..3 {
        assert!(gateway.request().starts_with("GET /health HTTP/1.1"));
        let request = gateway.request();
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        let body = request_body_json(&request);
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["max_tokens"], 128);
    }
    gateway.join();

    let result = root
        .join(".local")
        .join("benchmarks")
        .join("results")
        .join(format!("{benchmark_id}.json"));
    let payload: serde_json::Value =
        serde_json::from_slice(&fs::read(&result).expect("read benchmark result"))
            .expect("parse benchmark result");
    assert_eq!(payload["schema_version"], "1.2.0");
    assert_eq!(payload["workload"]["pp"], 64);
    assert_eq!(payload["workload"]["tg"], 16);
    assert_eq!(payload["workload"]["batch_size"], 64);
    assert_eq!(
        payload["runs"]["prefill_tps"],
        serde_json::json!([160.0, 160.0, 160.0])
    );
    assert_eq!(
        payload["runs"]["decode_tps"],
        serde_json::json!([20.0, 20.0, 20.0])
    );
    assert_eq!(payload["optimization"]["mode"], "baseline");
    assert_eq!(payload["optimization"]["methods"], serde_json::json!([]));
    let run_command = payload["runtime"]["run_command"]
        .as_str()
        .expect("runtime command is a string");
    assert!(run_command.contains("llama-server -m /models/qwen.gguf -b 64"));
    assert!(run_command.contains("--api-key"));
    assert!(run_command.contains("<redacted>"));
    assert!(!run_command.contains("runtime-secret"));

    let mut list = Command::cargo_bin("omniinfer").expect("binary exists");
    list.env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["bench", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains(benchmark_id))
        .stdout(predicate::str::contains("qwen3-5-2b"));
    fs::remove_dir_all(root).ok();
}

#[test]
fn bench_requires_an_explicit_optimization_declaration() {
    let mut command = Command::cargo_bin("omniinfer").expect("binary exists");
    command
        .args([
            "bench",
            "run",
            "--catalog-model-id",
            "qwen3-5-2b",
            "--format",
            "GGUF",
            "--quantization",
            "Q4_0",
            "--model-url",
            "https://example.com/qwen.gguf",
            "--device-name",
            "Test GPU",
            "--soc",
            "test-gpu",
            "--backend-version",
            "test-runtime-1",
            "--build-command",
            "bash build.sh",
            "--submitter-name",
            "OmniInfer Test",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Explicit optimization declaration is required",
        ));
}
