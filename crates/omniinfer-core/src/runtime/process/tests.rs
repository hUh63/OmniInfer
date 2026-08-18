use std::fs;
use std::net::TcpListener;

use super::*;

#[test]
fn accepts_empty_success_health_response() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            if line == "\r\n" || line == "\n" {
                break;
            }
            line.clear();
        }
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
    });
    assert!(health_endpoint_ready(
        "127.0.0.1",
        port,
        Duration::from_secs(1)
    ));
    handle.join().unwrap();
}

#[test]
fn starts_ready_process_and_stops_on_drop() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let root = temp_root("runtime-process-ready");
    let script = write_test_server(&root, port);
    let plan = ExternalRuntimePlan {
        command: test_script_command(&script),
        stop_command: None,
        cwd: root.clone(),
        port,
        ctx_size: None,
        log_file_name: "runtime.log".to_string(),
        proxy_model_ref: None,
        protocol: crate::runtime_plan::ExternalServerProtocol::LlamaCppServer,
        client_endpoint: format!("http://127.0.0.1:{port}"),
        readiness_probe: RuntimeReadinessProbe::HttpHealth,
    };
    let process = RuntimeProcess::start(
        &plan,
        RuntimeProcessOptions {
            log_path: root.join("runtime.log"),
            env: Vec::new(),
            startup_timeout: Duration::from_secs(5),
            health_host: "127.0.0.1".to_string(),
        },
    )
    .unwrap();
    assert!(process.info().pid > 0);
    let pid = process.info().pid;
    drop(process);
    assert!(process_exited(pid, Duration::from_secs(3)));
    assert!(TcpStream::connect(("127.0.0.1", port)).is_err());
    fs::remove_dir_all(root).ok();
}

#[cfg(unix)]
#[test]
fn early_exit_reaps_process_group_descendants() {
    let root = temp_root("runtime-process-group-early-exit");
    fs::create_dir_all(&root).unwrap();
    let script = root.join("spawn-child-and-exit.sh");
    fs::write(
        &script,
        "#!/usr/bin/env bash\nsleep 30 &\necho $! > child.pid\nexit 7\n",
    )
    .unwrap();
    make_executable(&script);
    let plan = ExternalRuntimePlan {
        command: test_script_command(&script),
        stop_command: None,
        cwd: root.clone(),
        port: 9,
        ctx_size: None,
        log_file_name: "runtime.log".to_string(),
        proxy_model_ref: None,
        protocol: crate::runtime_plan::ExternalServerProtocol::LlamaCppServer,
        client_endpoint: "http://127.0.0.1:9".to_string(),
        readiness_probe: RuntimeReadinessProbe::HttpHealth,
    };

    let error = RuntimeProcess::start(
        &plan,
        RuntimeProcessOptions {
            log_path: root.join("runtime.log"),
            env: Vec::new(),
            startup_timeout: Duration::from_secs(2),
            health_host: "127.0.0.1".to_string(),
        },
    )
    .unwrap_err();

    assert!(matches!(error, RuntimeProcessError::EarlyExit));
    let child_pid = fs::read_to_string(root.join("child.pid"))
        .unwrap()
        .trim()
        .parse::<u32>()
        .unwrap();
    assert!(process_exited(child_pid, Duration::from_secs(3)));
    fs::remove_dir_all(root).ok();
}

#[cfg(unix)]
#[test]
fn explicit_stop_does_not_run_stop_hook_again_on_drop() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let root = temp_root("runtime-process-idempotent-stop");
    let server = write_test_server(&root, port);
    let hook = root.join("count-stop.sh");
    let count = root.join("stop-count");
    fs::write(&hook, "#!/usr/bin/env bash\nprintf 'x' >> \"$1\"\n").unwrap();
    make_executable(&hook);
    let plan = ExternalRuntimePlan {
        command: test_script_command(&server),
        stop_command: Some(vec![
            "bash".to_string(),
            hook.display().to_string(),
            count.display().to_string(),
        ]),
        cwd: root.clone(),
        port,
        ctx_size: None,
        log_file_name: "runtime.log".to_string(),
        proxy_model_ref: None,
        protocol: crate::runtime_plan::ExternalServerProtocol::LlamaCppServer,
        client_endpoint: format!("http://127.0.0.1:{port}"),
        readiness_probe: RuntimeReadinessProbe::HttpHealth,
    };
    let start_started = Instant::now();
    let mut process = RuntimeProcess::start(
        &plan,
        RuntimeProcessOptions {
            log_path: root.join("runtime.log"),
            env: Vec::new(),
            startup_timeout: Duration::from_secs(5),
            health_host: "127.0.0.1".to_string(),
        },
    )
    .unwrap();
    assert!(
        start_started.elapsed() < Duration::from_secs(10),
        "runtime start must honor the readiness timeout"
    );

    let stop_started = Instant::now();
    process.stop(Duration::from_secs(2)).unwrap();
    assert!(
        stop_started.elapsed() < Duration::from_secs(10),
        "runtime stop must not block on diagnostic log durability"
    );
    drop(process);

    assert_eq!(fs::read_to_string(count).unwrap(), "x");
    assert!(
        fs::read_to_string(root.join("runtime.log"))
            .unwrap()
            .contains("fixture ready"),
        "runtime logs must remain readable after normal handle close"
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn returns_early_exit_for_failed_process() {
    let root = temp_root("runtime-process-fail");
    let script = write_failed_process(&root);
    let plan = ExternalRuntimePlan {
        command: test_script_command(&script),
        stop_command: None,
        cwd: root.clone(),
        port: 9,
        ctx_size: None,
        log_file_name: "runtime.log".to_string(),
        proxy_model_ref: None,
        protocol: crate::runtime_plan::ExternalServerProtocol::LlamaCppServer,
        client_endpoint: "http://127.0.0.1:9".to_string(),
        readiness_probe: RuntimeReadinessProbe::HttpHealth,
    };
    let error = RuntimeProcess::start(
        &plan,
        RuntimeProcessOptions {
            log_path: root.join("runtime.log"),
            env: Vec::new(),
            startup_timeout: Duration::from_secs(1),
            health_host: "127.0.0.1".to_string(),
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, RuntimeProcessError::EarlyExit),
        "unexpected error: {error:?}"
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn drop_kills_unready_process() {
    let root = temp_root("runtime-process-unready");
    let script = write_sleep_process(&root);
    let plan = ExternalRuntimePlan {
        command: test_script_command(&script),
        stop_command: None,
        cwd: root.clone(),
        port: 9,
        ctx_size: None,
        log_file_name: "runtime.log".to_string(),
        proxy_model_ref: None,
        protocol: crate::runtime_plan::ExternalServerProtocol::LlamaCppServer,
        client_endpoint: "http://127.0.0.1:9".to_string(),
        readiness_probe: RuntimeReadinessProbe::HttpHealth,
    };
    let error = RuntimeProcess::start(
        &plan,
        RuntimeProcessOptions {
            log_path: root.join("runtime.log"),
            env: Vec::new(),
            startup_timeout: Duration::from_millis(250),
            health_host: "127.0.0.1".to_string(),
        },
    )
    .unwrap_err();
    assert!(matches!(error, RuntimeProcessError::ReadyTimeout));
    fs::remove_dir_all(root).ok();
}

#[test]
fn tcp_readiness_ignores_stale_log_marker_and_occupied_port() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let root = temp_root("runtime-process-stale-marker");
    fs::create_dir_all(&root).unwrap();
    let script = write_sleep_process(&root);
    let log_path = root.join("runtime.log");
    let marker = format!("vla-server: bound to tcp://127.0.0.1:{port}. ready.");
    fs::write(&log_path, format!("{marker}\n")).unwrap();
    let plan = ExternalRuntimePlan {
        command: test_script_command(&script),
        stop_command: None,
        cwd: root.clone(),
        port,
        ctx_size: None,
        log_file_name: "runtime.log".to_string(),
        proxy_model_ref: None,
        protocol: crate::runtime_plan::ExternalServerProtocol::VlaCppZmqServer,
        client_endpoint: format!("tcp://127.0.0.1:{port}"),
        readiness_probe: RuntimeReadinessProbe::TcpConnectAndLog { marker },
    };

    let error = RuntimeProcess::start(
        &plan,
        RuntimeProcessOptions {
            log_path,
            env: Vec::new(),
            startup_timeout: Duration::from_millis(250),
            health_host: "127.0.0.1".to_string(),
        },
    )
    .unwrap_err();
    assert!(matches!(error, RuntimeProcessError::ReadyTimeout));

    drop(listener);
    fs::remove_dir_all(root).ok();
}

#[test]
fn stop_hook_reports_failure_and_timeout() {
    let root = temp_root("runtime-stop-hook");
    let hook = write_stop_hook(&root);

    let mut success = test_script_command(&hook);
    success.push("success".to_string());
    run_stop_hook(&success, Duration::from_secs(1)).unwrap();

    let mut failure = test_script_command(&hook);
    failure.push("failure".to_string());
    let error = run_stop_hook(&failure, Duration::from_secs(1)).unwrap_err();
    assert!(matches!(error, RuntimeProcessError::StopHook(_)));
    assert!(error.to_string().contains("injected stop failure"));

    let mut timeout = test_script_command(&hook);
    timeout.push("timeout".to_string());
    let error = run_stop_hook(&timeout, Duration::from_millis(100)).unwrap_err();
    assert!(matches!(error, RuntimeProcessError::StopHook(_)));
    assert!(error.to_string().contains("timed out"));

    fs::remove_dir_all(root).ok();
}

#[test]
fn failed_stop_hook_still_reaps_wrapper_process() {
    let root = temp_root("runtime-stop-hook-reap");
    let sleep = write_sleep_process(&root);
    let hook = write_stop_hook(&root);
    let sleep_command = test_script_command(&sleep);
    let mut child = Command::new(&sleep_command[0])
        .args(&sleep_command[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut failure = test_script_command(&hook);
    failure.push("failure".to_string());

    let error =
        terminate_runtime(&mut child, Some(&failure), Duration::from_millis(250)).unwrap_err();
    assert!(matches!(error, RuntimeProcessError::StopHook(_)));
    assert!(child.try_wait().unwrap().is_some());

    fs::remove_dir_all(root).ok();
}

fn write_test_server(root: &Path, port: u16) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    #[cfg(any(windows, target_os = "macos"))]
    {
        let executable = root.join(if cfg!(windows) {
            "server.exe"
        } else {
            "server"
        });
        compile_test_exe(
            root,
            "server.rs",
            &executable,
            &format!(
                r##"
use std::io::{{BufRead, BufReader, Write}};
use std::net::{{TcpListener, TcpStream}};

fn main() {{
    let listener = TcpListener::bind("127.0.0.1:{port}").unwrap();
    println!("fixture ready");
    std::io::stdout().flush().unwrap();
    for stream in listener.incoming().flatten() {{
        handle(stream);
    }}
}}

fn handle(mut stream: TcpStream) {{
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {{
        return;
    }}
    loop {{
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {{
            return;
        }}
        if line == "\r\n" || line == "\n" || line.is_empty() {{
            break;
        }}
    }}
    let body = r#"{{"status":"ok"}}"#;
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {{}}\r\nConnection: close\r\n\r\n",
        body.as_bytes().len()
    );
    let _ = stream.write_all(headers.as_bytes());
    let _ = stream.write_all(body.as_bytes());
}}
"##
            ),
        );
        return executable;
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        let script = root.join("server.sh");
        fs::write(
            &script,
            format!(
                r#"#!/usr/bin/env bash
exec python3 - <<'PY'
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        raw = json.dumps({{"status": "ok"}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

print("fixture ready", flush=True)
HTTPServer(("127.0.0.1", {port}), Handler).serve_forever()
PY
"#
            ),
        )
        .unwrap();
        make_executable(&script);
        script
    }
}

fn write_failed_process(root: &Path) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    #[cfg(windows)]
    {
        let executable = root.join("fail.exe");
        compile_test_exe(
            root,
            "fail.rs",
            &executable,
            "fn main() { std::process::exit(7); }\n",
        );
        executable
    }
    #[cfg(not(windows))]
    {
        let script = root.join("fail.sh");
        fs::write(&script, "#!/usr/bin/env bash\nexit 7\n").unwrap();
        make_executable(&script);
        script
    }
}

fn write_sleep_process(root: &Path) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    #[cfg(windows)]
    {
        let executable = root.join("sleep.exe");
        compile_test_exe(
            root,
            "sleep.rs",
            &executable,
            r#"
fn main() {
    std::thread::sleep(std::time::Duration::from_secs(30));
}
"#,
        );
        executable
    }
    #[cfg(not(windows))]
    {
        let script = root.join("sleep.sh");
        fs::write(&script, "#!/usr/bin/env bash\nexec sleep 30\n").unwrap();
        make_executable(&script);
        script
    }
}

fn write_stop_hook(root: &Path) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    #[cfg(windows)]
    {
        let executable = root.join("stop-hook.exe");
        compile_test_exe(
            root,
            "stop-hook.rs",
            &executable,
            r#"
fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("success") => {}
        Some("failure") => {
            eprintln!("injected stop failure");
            std::process::exit(9);
        }
        Some("timeout") => std::thread::sleep(std::time::Duration::from_secs(30)),
        _ => std::process::exit(2),
    }
}
"#,
        );
        executable
    }
    #[cfg(not(windows))]
    {
        let script = root.join("stop-hook.sh");
        fs::write(
                &script,
                "#!/usr/bin/env bash\ncase \"$1\" in success) exit 0;; failure) echo 'injected stop failure' >&2; exit 9;; timeout) exec sleep 30;; *) exit 2;; esac\n",
            )
            .unwrap();
        make_executable(&script);
        script
    }
}

#[cfg(any(windows, target_os = "macos"))]
fn compile_test_exe(root: &Path, source_name: &str, executable: &Path, code: &str) {
    let source = root.join(source_name);
    fs::write(&source, code).unwrap();
    let status = Command::new("rustc")
        .arg("--edition=2021")
        .arg(&source)
        .arg("-o")
        .arg(executable)
        .status()
        .expect("compile native test process");
    assert!(status.success(), "failed to compile native test process");
}

fn temp_root(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("omniinfer-{name}-{nanos}"))
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

#[cfg(all(unix, not(target_os = "macos")))]
fn test_script_command(path: &Path) -> Vec<String> {
    vec!["bash".to_string(), path.display().to_string()]
}

#[cfg(any(windows, target_os = "macos"))]
fn test_script_command(path: &Path) -> Vec<String> {
    vec![path.display().to_string()]
}

fn process_exited(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !process_exists(pid) {
            return true;
        }
        thread::sleep(Duration::from_millis(50));
    }
    false
}

#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

#[cfg(windows)]
fn process_exists(pid: u32) -> bool {
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}")])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}
