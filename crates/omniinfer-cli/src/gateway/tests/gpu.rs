use super::*;

#[test]
fn cuda_picker_prefers_lowest_idle_index() {
    let mut devices = parse_cuda_gpu_rows(
        "\
0, GPU-a, 128
1, GPU-b, 64
2, GPU-c, 0
",
    );
    apply_cuda_process_rows(&mut devices, "GPU-a, 1001\n");

    let choice = select_cuda_device_from_usage(&devices).unwrap();

    assert_eq!(choice.index, "1");
    assert_eq!(choice.warning, None);
}

#[test]
fn cuda_picker_warns_and_uses_least_loaded_when_all_busy() {
    let mut devices = parse_cuda_gpu_rows(
        "\
0, GPU-a, 900
1, GPU-b, 256
2, GPU-c, 512
",
    );
    apply_cuda_process_rows(&mut devices, "GPU-a, 1001\nGPU-b, 1002\nGPU-c, 1003\n");

    let choice = select_cuda_device_from_usage(&devices).unwrap();

    assert_eq!(choice.index, "1");
    assert!(
        choice
            .warning
            .as_deref()
            .unwrap()
            .contains("all CUDA GPUs appear to be in use")
    );
}

#[test]
fn cuda_picker_allows_driver_memory_when_no_compute_process() {
    let mut devices = parse_cuda_gpu_rows(
        "\
0, GPU-a, 512
1, GPU-b, 128
",
    );
    apply_cuda_process_rows(&mut devices, "GPU-a, 1001\n");

    let choice = select_cuda_device_from_usage(&devices).unwrap();

    assert_eq!(choice.index, "1");
    assert_eq!(choice.warning, None);
}

#[test]
fn cuda_picker_detects_explicit_multi_gpu_args() {
    assert!(uses_explicit_cuda_device_args(&[
        "--tensor-split".to_string(),
        "1,1".to_string()
    ]));
    assert!(uses_explicit_cuda_device_args(&[
        "--main-gpu=1".to_string(),
        "-ngl".to_string(),
        "999".to_string()
    ]));
    assert!(!uses_explicit_cuda_device_args(&[
        "-ngl".to_string(),
        "999".to_string()
    ]));
}

#[test]
fn gpu_status_parses_devices_and_owners() {
    let mut devices = parse_gpu_status_rows(
        "\
0, GPU-a, NVIDIA GeForce RTX 3090, 24576, 12000, 12576, 91
1, GPU-b, NVIDIA GeForce RTX 3090, 24576, 1, 24258, 0
",
    );
    let loaded = vec![LoadedRuntimeSummary {
        id: "qwen3.5-35b-a3b-q4_k_m".to_string(),
        owner_admin_id: Some("adminA".to_string()),
        backend_pid: 4242,
    }];
    let processes = parse_gpu_process_rows(
        "\
GPU-a, 4242, llama-server, 11998
GPU-a, 5151, python, 256
",
        &loaded,
    );

    apply_gpu_process_rows(&mut devices, processes);

    assert_eq!(devices.len(), 2);
    assert_eq!(devices[0].memory_total_mib, 24576);
    assert_eq!(devices[0].utilization_gpu_percent, Some(91));
    assert_eq!(devices[0].processes.len(), 2);
    assert_eq!(
        devices[0].processes[0].owner_model.as_deref(),
        Some("qwen3.5-35b-a3b-q4_k_m")
    );
    assert_eq!(
        devices[0].processes[0].owner_admin_id.as_deref(),
        Some("adminA")
    );
    assert_eq!(devices[0].processes[0].owner_type, "admin");
    assert_eq!(
        devices[0].processes[0].owner_name.as_deref(),
        Some("adminA")
    );
    assert_eq!(devices[0].processes[0].display_name, "llama-server");
    assert_eq!(devices[0].processes[1].owner_model, None);
    assert_eq!(devices[0].processes[1].owner_type, "user");
    assert!(devices[1].processes.is_empty());
}

#[test]
fn gpu_status_payload_uses_numeric_indexes() {
    let device = GpuStatusDevice {
        index: "7".to_string(),
        uuid: "GPU-x".to_string(),
        name: "NVIDIA GeForce RTX 3090".to_string(),
        memory_total_mib: 24576,
        memory_used_mib: 1,
        memory_free_mib: 24258,
        utilization_gpu_percent: Some(0),
        processes: Vec::new(),
    };

    let payload = gpu_status_device_payload(&device);

    assert_eq!(payload["index"], json!(7));
    assert_eq!(payload["memory_total_mb"], json!(24576));
    assert!(payload["processes"].as_array().unwrap().is_empty());
}
