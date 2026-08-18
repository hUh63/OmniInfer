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
