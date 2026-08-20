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
fn visual_projector_does_not_change_model_context_components() {
    let root = std::env::temp_dir().join(format!(
        "omniinfer-resource-formula-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&root).unwrap();
    let model = root.join("model.gguf");
    let projector = root.join("mmproj.gguf");
    fs::File::create(&model).unwrap().set_len(5 * GIB).unwrap();
    fs::File::create(&projector).unwrap().set_len(GIB).unwrap();
    let backend = BackendRegistry::load_current()
        .get("llama.cpp-linux-cuda")
        .cloned()
        .unwrap_or_else(|| backend_registry::BackendSpec {
            id: "llama.cpp-linux-cuda".to_string(),
            label: "test".to_string(),
            family: "llama.cpp".to_string(),
            runtime_dir: root.display().to_string(),
            launcher_path: None,
            models_dir: None,
            catalog_url: None,
            description: "test".to_string(),
            capabilities: vec!["cuda".to_string()],
            default_args: Vec::new(),
            runtime_mode: "external_server".to_string(),
            model_artifact: "gguf".to_string(),
            supports_mmproj: true,
            supports_ctx_size: true,
            python_modules: Vec::new(),
            external_server_protocol: Some("llama_cpp".to_string()),
            log_file_name: "test.log".to_string(),
        });
    let text = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        None,
        2048,
        Some("0"),
        false,
    )
    .unwrap();
    let visual = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        Some(projector.to_str().unwrap()),
        2048,
        Some("0"),
        false,
    )
    .unwrap();
    for name in ["kv_cache", "activation"] {
        let text_bytes = text
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap()
            .bytes;
        let visual_bytes = visual
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap()
            .bytes;
        assert_eq!(text_bytes, visual_bytes, "component {name}");
    }
    assert!(
        visual
            .components()
            .iter()
            .any(|component| component.name == "mmproj")
    );
    for name in ["framework_overhead", "allocator_slack"] {
        let text_bytes = text
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap()
            .bytes;
        let visual_bytes = visual
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap()
            .bytes;
        assert!(visual_bytes > text_bytes, "component {name}");
    }
    assert!(
        visual.domains()[&MemoryDomain::Cuda("0".to_string())]
            > text.domains()[&MemoryDomain::Cuda("0".to_string())]
    );
    fs::remove_dir_all(root).ok();
}

fn speculative_test_backend(id: &str, family: &str, cuda: bool) -> backend_registry::BackendSpec {
    backend_registry::BackendSpec {
        id: id.to_string(),
        label: "test".to_string(),
        family: family.to_string(),
        runtime_dir: String::new(),
        launcher_path: None,
        models_dir: None,
        catalog_url: None,
        description: "test".to_string(),
        capabilities: cuda.then(|| "cuda".to_string()).into_iter().collect(),
        default_args: Vec::new(),
        runtime_mode: "external_server".to_string(),
        model_artifact: "gguf".to_string(),
        supports_mmproj: true,
        supports_ctx_size: true,
        python_modules: Vec::new(),
        external_server_protocol: Some("llama_cpp".to_string()),
        log_file_name: "test.log".to_string(),
    }
}

fn speculative_snapshot(
    capacity: u64,
    reserved: u64,
    committed: u64,
) -> omniinfer_core::resource_ledger::ResourceLedgerSnapshot {
    omniinfer_core::resource_ledger::ResourceLedgerSnapshot {
        capacity_snapshot_id: 1,
        capacities: BTreeMap::from([(MemoryDomain::Cuda("0".to_string()), capacity)]),
        reserved: BTreeMap::from([(MemoryDomain::Cuda("0".to_string()), reserved)])
            .into_iter()
            .filter(|(_, bytes)| *bytes > 0)
            .collect(),
        committed: BTreeMap::from([(MemoryDomain::Cuda("0".to_string()), committed)])
            .into_iter()
            .filter(|(_, bytes)| *bytes > 0)
            .collect(),
    }
}

fn speculative_budget(estimated: u64, slack: u64) -> ResourceBudget {
    ResourceBudget::from_components(vec![
        BudgetComponent {
            name: "model".to_string(),
            domain: MemoryDomain::Cuda("0".to_string()),
            bytes: estimated - slack,
        },
        BudgetComponent {
            name: "allocator_slack".to_string(),
            domain: MemoryDomain::Cuda("0".to_string()),
            bytes: slack,
        },
    ])
    .unwrap()
}

#[test]
fn speculative_cuda_admission_enforces_narrow_boundaries() {
    let backend = speculative_test_backend("llama.cpp-linux-cuda", "llama.cpp", true);
    let budget = speculative_budget(1_000, 100);
    let accepted = [json!({}), json!({"mmproj": "/projector.gguf"})];
    for payload in accepted {
        let decision = speculative_reservation(
            &backend,
            &payload,
            &budget,
            false,
            Some(speculative_snapshot(900, 0, 0)),
        )
        .unwrap()
        .unwrap();
        assert_eq!(decision.available, 900);
        assert_eq!(decision.shortfall, 100);
        assert_eq!(decision.waived_slack, decision.shortfall);
        assert_eq!(
            decision.budget.domains()[&MemoryDomain::Cuda("0".to_string())],
            900
        );
    }

    let less_than_slack = speculative_reservation(
        &backend,
        &json!({}),
        &budget,
        false,
        Some(speculative_snapshot(950, 0, 0)),
    )
    .unwrap()
    .unwrap();
    assert_eq!(less_than_slack.shortfall, 50);
    assert_eq!(less_than_slack.waived_slack, 50);

    for (available, payload, replicate) in [
        (899, json!({}), false),
        (1_000 - 100 - 1, json!({}), false),
        (900, json!({"resource_budget_bytes": 1}), false),
        (900, json!({}), true),
    ] {
        assert!(
            speculative_reservation(
                &backend,
                &payload,
                &budget,
                replicate,
                Some(speculative_snapshot(available, 0, 0)),
            )
            .unwrap()
            .is_none()
        );
    }

    let oversized_slack = speculative_budget(
        3 * SPECULATIVE_ALLOCATOR_SLACK_LIMIT,
        2 * SPECULATIVE_ALLOCATOR_SLACK_LIMIT,
    );
    assert!(
        speculative_reservation(
            &backend,
            &json!({}),
            &oversized_slack,
            false,
            Some(speculative_snapshot(
                2 * SPECULATIVE_ALLOCATOR_SLACK_LIMIT - 1,
                0,
                0
            )),
        )
        .unwrap()
        .is_none()
    );

    for (candidate, is_cuda, id, family, reserved, committed) in [
        (Some(900), false, "llama.cpp-linux-cuda", "llama.cpp", 0, 0),
        (Some(900), true, "other-cuda", "other", 0, 0),
        (Some(900), true, "llama.cpp-linux-cuda", "llama.cpp", 1, 0),
        (Some(900), true, "llama.cpp-linux-cuda", "llama.cpp", 0, 1),
    ] {
        let backend = speculative_test_backend(id, family, is_cuda);
        assert!(
            speculative_reservation(
                &backend,
                &json!({}),
                &budget,
                false,
                Some(speculative_snapshot(
                    candidate.unwrap(),
                    reserved,
                    committed
                )),
            )
            .unwrap()
            .is_none()
        );
    }

    let multi = ResourceBudget::from_components(vec![
        BudgetComponent {
            name: "model".to_string(),
            domain: MemoryDomain::Cuda("0".to_string()),
            bytes: 900,
        },
        BudgetComponent {
            name: "model".to_string(),
            domain: MemoryDomain::Cuda("1".to_string()),
            bytes: 900,
        },
    ])
    .unwrap();
    assert!(
        speculative_reservation(
            &backend,
            &json!({}),
            &multi,
            false,
            Some(speculative_snapshot(900, 0, 0)),
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn speculative_reservation_is_exclusive_and_rolls_back() {
    let backend = speculative_test_backend("llama.cpp-linux-cuda", "llama.cpp", true);
    let estimated = 1_000;
    let available = 900;
    let budget = speculative_budget(estimated, 100);
    let decision = speculative_reservation(
        &backend,
        &json!({}),
        &budget,
        false,
        Some(speculative_snapshot(available, 0, 0)),
    )
    .unwrap()
    .unwrap();
    let capacity = ResourceCapacity::new(
        1,
        BTreeMap::from([(MemoryDomain::Cuda("0".to_string()), available)]),
    )
    .unwrap();
    let mut ledger = ResourceLedger::new(capacity);
    let reservation = ledger
        .reserve("speculative", decision.budget.clone())
        .unwrap();
    assert!(ledger.reserve("second", decision.budget.clone()).is_err());
    assert!(ledger.rollback(reservation));
    assert!(ledger.reserve("second", decision.budget).is_ok());
}

#[test]
fn speculative_admission_payload_is_deterministic_and_separate_from_cuda_warning() {
    assert_eq!(speculative_admission_payload(None), Value::Null);
    let admission = SpeculativeAdmission {
        device: "0".to_string(),
        estimated: 1_000,
        exclusive: 900,
        shortfall: 100,
        waived_allocator_slack: 100,
    };
    assert_eq!(
        speculative_admission_payload(Some(&admission)),
        json!({
            "speculative": true,
            "device": "0",
            "estimated_cuda_bytes": 1_000,
            "exclusive_reservation_bytes": 900,
            "shortfall_bytes": 100,
            "waived_allocator_slack_bytes": 100,
        })
    );
    let smaller_shortfall = SpeculativeAdmission {
        waived_allocator_slack: 37,
        shortfall: 37,
        ..admission
    };
    assert_eq!(
        speculative_admission_payload(Some(&smaller_shortfall))["waived_allocator_slack_bytes"],
        37
    );
}

#[test]
fn speculative_domain_exclusivity_survives_refresh_and_releases_by_owner() {
    let capacity = ResourceCapacity::new(
        1,
        BTreeMap::from([
            (MemoryDomain::Cuda("0".to_string()), 1024 * GIB),
            (MemoryDomain::Cuda("1".to_string()), 1024 * GIB),
        ]),
    )
    .unwrap();
    let mut ledger = ResourceLedger::new(capacity);
    let owner_reservation = ledger
        .reserve(
            "speculative-owner",
            ResourceBudget::from_components(vec![BudgetComponent {
                name: "owner".to_string(),
                domain: MemoryDomain::Cuda("0".to_string()),
                bytes: 1,
            }])
            .unwrap(),
        )
        .unwrap();
    let owner_allocation = ledger.commit(owner_reservation).unwrap();
    let mut manager = RustRuntimeManager {
        resource_ledger: Some(ledger),
        speculative_domains: BTreeMap::from([(
            MemoryDomain::Cuda("0".to_string()),
            owner_allocation,
        )]),
        next_capacity_snapshot: 2,
        ..Default::default()
    };
    let cuda0 = ResourceBudget::from_components(vec![BudgetComponent {
        name: "follow_on".to_string(),
        domain: MemoryDomain::Cuda("0".to_string()),
        bytes: 1,
    }])
    .unwrap();
    let error = manager
        .reserve_runtime_resources("same-device", &cuda0, Some("0"))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exclusively held by a speculative runtime"),
        "{error:#}"
    );
    // Model-key promotion/reuse cannot clear an owner: cleanup is allocation-identity based.
    let wrong_reservation = manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .reserve(
            "reused-old-key",
            ResourceBudget::from_components(vec![BudgetComponent {
                name: "other".to_string(),
                domain: MemoryDomain::Cuda("1".to_string()),
                bytes: 1,
            }])
            .unwrap(),
        )
        .unwrap();
    let wrong_allocation = manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .commit(wrong_reservation)
        .unwrap();
    manager.clear_speculative_owner(wrong_allocation);
    assert!(
        manager
            .speculative_domains
            .contains_key(&MemoryDomain::Cuda("0".to_string()))
    );
    manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .release(wrong_allocation);

    let cuda1 = ResourceBudget::from_components(vec![BudgetComponent {
        name: "other-device".to_string(),
        domain: MemoryDomain::Cuda("1".to_string()),
        bytes: 1,
    }])
    .unwrap();
    let other_reservation = manager
        .reserve_runtime_resources("other-device", &cuda1, Some("1"))
        .unwrap();
    manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .rollback(other_reservation);

    manager.clear_speculative_owner(owner_allocation);
    let released_reservation = manager
        .reserve_runtime_resources("after-release", &cuda0, Some("0"))
        .unwrap();
    assert!(
        manager
            .resource_ledger
            .as_mut()
            .unwrap()
            .rollback(released_reservation)
    );
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
