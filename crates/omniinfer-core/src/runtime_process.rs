use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::runtime_plan::{ExternalRuntimePlan, RuntimeReadinessProbe};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeProcessOptions {
    pub log_path: PathBuf,
    pub env: Vec<(String, String)>,
    pub startup_timeout: Duration,
    pub health_host: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeProcessInfo {
    pub pid: u32,
    pub port: u16,
    pub command: Vec<String>,
    pub log_path: PathBuf,
}

#[derive(Debug)]
pub struct RuntimeProcess {
    child: Child,
    stop_command: Option<Vec<String>>,
    stopped: bool,
    info: RuntimeProcessInfo,
}

#[derive(Debug, Error)]
pub enum RuntimeProcessError {
    #[error("runtime command is empty")]
    EmptyCommand,
    #[error("failed to create runtime log directory {path}: {source}")]
    CreateLogDir {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open runtime log {path}: {source}")]
    OpenLog {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to duplicate runtime log handle {path}: {source}")]
    CloneLog {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to spawn runtime process: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("runtime exited before becoming ready")]
    EarlyExit,
    #[error("runtime did not become ready in time")]
    ReadyTimeout,
    #[error("runtime stop hook failed: {0}")]
    StopHook(String),
}

impl RuntimeProcess {
    pub fn start(
        plan: &ExternalRuntimePlan,
        options: RuntimeProcessOptions,
    ) -> Result<Self, RuntimeProcessError> {
        let executable = plan
            .command
            .first()
            .ok_or(RuntimeProcessError::EmptyCommand)?;
        if let Some(parent) = options.log_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                RuntimeProcessError::CreateLogDir {
                    path: parent.display().to_string(),
                    source,
                }
            })?;
        }
        let log_handle = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&options.log_path)
            .map_err(|source| RuntimeProcessError::OpenLog {
                path: options.log_path.display().to_string(),
                source,
            })?;
        let log_start_offset = log_handle
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let stderr = log_handle
            .try_clone()
            .map_err(|source| RuntimeProcessError::CloneLog {
                path: options.log_path.display().to_string(),
                source,
            })?;
        let mut command = Command::new(executable);
        command
            .args(plan.command.iter().skip(1))
            .current_dir(&plan.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_handle))
            .stderr(Stdio::from(stderr));
        for (key, value) in &options.env {
            command.env(key, value);
        }
        isolate_process_tree(&mut command);
        hide_child_window(&mut command);
        let mut child = command.spawn()?;
        let readiness = wait_runtime_ready(
            &options.health_host,
            plan.port,
            &plan.readiness_probe,
            options.startup_timeout,
            &mut child,
            &options.log_path,
            log_start_offset,
        );
        match readiness {
            Ok(true) => {}
            Ok(false) => {
                let _ = terminate_runtime(
                    &mut child,
                    plan.stop_command.as_deref(),
                    Duration::from_secs(2),
                );
                return Err(RuntimeProcessError::ReadyTimeout);
            }
            Err(error) => {
                let _ = terminate_runtime(
                    &mut child,
                    plan.stop_command.as_deref(),
                    Duration::from_secs(2),
                );
                return Err(error);
            }
        }
        let info = RuntimeProcessInfo {
            pid: child.id(),
            port: plan.port,
            command: plan.command.clone(),
            log_path: options.log_path,
        };
        Ok(Self {
            child,
            stop_command: plan.stop_command.clone(),
            stopped: false,
            info,
        })
    }

    pub fn info(&self) -> &RuntimeProcessInfo {
        &self.info
    }

    pub fn has_exited(&mut self) -> Result<bool, RuntimeProcessError> {
        Ok(self.child.try_wait()?.is_some())
    }

    pub fn stop(&mut self, grace: Duration) -> Result<(), RuntimeProcessError> {
        if self.stopped {
            return Ok(());
        }
        let result = terminate_runtime(&mut self.child, self.stop_command.as_deref(), grace);
        if self.child.try_wait().ok().flatten().is_some() {
            self.stopped = true;
        }
        // The child owns the cloned log descriptors, which close when it exits.
        // Diagnostic logs do not require a blocking durability fsync on every stop.
        result
    }
}

impl Drop for RuntimeProcess {
    fn drop(&mut self) {
        let _ = self.stop(Duration::from_secs(1));
    }
}

fn wait_runtime_ready(
    host: &str,
    port: u16,
    probe: &RuntimeReadinessProbe,
    timeout: Duration,
    child: &mut Child,
    log_path: &Path,
    log_start_offset: u64,
) -> Result<bool, RuntimeProcessError> {
    let deadline = Instant::now() + timeout;
    let mut log_cursor = log_start_offset;
    let mut log_tail = Vec::new();
    let mut log_marker_seen = false;
    while Instant::now() < deadline {
        if child.try_wait()?.is_some() {
            return Err(RuntimeProcessError::EarlyExit);
        }
        let ready = match probe {
            RuntimeReadinessProbe::HttpHealth => {
                health_endpoint_ready(host, port, Duration::from_millis(500))
            }
            RuntimeReadinessProbe::TcpConnectAndLog { marker } => {
                log_marker_seen |= appended_log_contains(
                    log_path,
                    &mut log_cursor,
                    &mut log_tail,
                    marker.as_bytes(),
                );
                log_marker_seen && tcp_endpoint_ready(host, port, Duration::from_millis(500))
            }
        };
        if ready {
            return Ok(true);
        }
        thread::sleep(Duration::from_millis(100));
    }
    if child.try_wait()?.is_some() {
        return Err(RuntimeProcessError::EarlyExit);
    }
    Ok(false)
}

fn appended_log_contains(path: &Path, cursor: &mut u64, tail: &mut Vec<u8>, marker: &[u8]) -> bool {
    if marker.is_empty() {
        return true;
    }
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    if file.seek(SeekFrom::Start(*cursor)).is_err() {
        return false;
    }
    let mut appended = Vec::new();
    if file.take(1024 * 1024).read_to_end(&mut appended).is_err() {
        return false;
    }
    *cursor = cursor.saturating_add(appended.len() as u64);
    tail.extend_from_slice(&appended);
    let found = tail
        .windows(marker.len())
        .any(|candidate| candidate == marker);
    if !found {
        let keep = usize::min(marker.len().saturating_sub(1), tail.len());
        if keep == 0 {
            tail.clear();
        } else {
            tail.drain(..tail.len() - keep);
        }
    }
    found
}

fn health_endpoint_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let Some(mut stream) = connect_endpoint(host, port, timeout) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let request =
        format!("GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    if reader.read_line(&mut status_line).is_err() {
        return false;
    }
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| (200..300).contains(&status))
}

fn tcp_endpoint_ready(host: &str, port: u16, timeout: Duration) -> bool {
    connect_endpoint(host, port, timeout).is_some()
}

fn connect_endpoint(host: &str, port: u16, timeout: Duration) -> Option<TcpStream> {
    let Ok(addrs) = (host, port).to_socket_addrs() else {
        return None;
    };
    for addr in addrs {
        if let Ok(stream) = TcpStream::connect_timeout(&addr, timeout) {
            return Some(stream);
        }
    }
    None
}

fn terminate_child(child: &mut Child, grace: Duration) -> Result<(), RuntimeProcessError> {
    #[cfg(unix)]
    {
        let pid = child.id();
        let _ = child.try_wait()?;
        signal_process_group(pid, "-TERM");
        let deadline = Instant::now() + grace;
        while Instant::now() < deadline {
            let child_exited = child.try_wait()?.is_some();
            if child_exited && !process_group_exists(pid) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(50));
        }
        signal_process_group(pid, "-KILL");
        if child.try_wait()?.is_none() {
            child.kill()?;
        }
        let _ = child.wait();
        Ok(())
    }

    #[cfg(not(unix))]
    {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        terminate_process(child.id());
        let deadline = Instant::now() + grace;
        while Instant::now() < deadline {
            if child.try_wait()?.is_some() {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(50));
        }
        child.kill()?;
        let _ = child.wait();
        Ok(())
    }
}

fn terminate_runtime(
    child: &mut Child,
    stop_command: Option<&[String]>,
    grace: Duration,
) -> Result<(), RuntimeProcessError> {
    let hook_result = stop_command
        .map(|command| run_stop_hook(command, grace.min(Duration::from_secs(5))))
        .transpose();
    let child_result = terminate_child(child, grace);
    hook_result?;
    child_result
}

fn run_stop_hook(command: &[String], timeout: Duration) -> Result<(), RuntimeProcessError> {
    let Some(executable) = command.first() else {
        return Err(RuntimeProcessError::StopHook(
            "stop command is empty".to_string(),
        ));
    };
    let mut process = Command::new(executable);
    process
        .args(command.iter().skip(1))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    hide_child_window(&mut process);
    let mut hook = process.spawn()?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = hook.try_wait()? {
            if status.success() {
                return Ok(());
            }
            let mut stderr = String::new();
            if let Some(mut stream) = hook.stderr.take() {
                let _ = stream.read_to_string(&mut stderr);
            }
            let detail = stderr.trim();
            return Err(RuntimeProcessError::StopHook(if detail.is_empty() {
                format!("command exited with {status}")
            } else {
                detail.to_string()
            }));
        }
        if Instant::now() >= deadline {
            let _ = hook.kill();
            let _ = hook.wait();
            return Err(RuntimeProcessError::StopHook(
                "stop command timed out".to_string(),
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

#[cfg(unix)]
fn signal_process_group(pid: u32, signal: &str) {
    let signal = match signal {
        "-TERM" => libc::SIGTERM,
        "-KILL" => libc::SIGKILL,
        _ => return,
    };
    let Some(process_group) = process_group_id(pid) else {
        return;
    };
    // SAFETY: kill(2) does not dereference pointers. A negative PID targets
    // the process group created for this child by isolate_process_tree().
    unsafe {
        libc::kill(-process_group, signal);
    }
}

#[cfg(unix)]
fn process_group_exists(pid: u32) -> bool {
    let Some(process_group) = process_group_id(pid) else {
        return false;
    };
    // SAFETY: signal 0 only checks for the process group's existence and
    // permissions; it does not deliver a signal or dereference pointers.
    if unsafe { libc::kill(-process_group, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(unix)]
fn process_group_id(pid: u32) -> Option<libc::pid_t> {
    libc::pid_t::try_from(pid).ok().filter(|pid| *pid > 0)
}

#[cfg(windows)]
fn terminate_process(pid: u32) {
    let mut command = Command::new("taskkill");
    hide_child_window(&mut command);
    let _ = command
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn hide_child_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}

fn isolate_process_tree(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(not(unix))]
    {
        let _ = command;
    }
}

#[allow(dead_code)]
fn _path_exists(path: &Path) -> bool {
    path.exists()
}

#[cfg(test)]
mod tests {
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
        #[cfg(windows)]
        {
            let executable = root.join("server.exe");
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
        #[cfg(not(windows))]
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

    #[cfg(windows)]
    fn compile_test_exe(root: &Path, source_name: &str, executable: &Path, code: &str) {
        let source = root.join(source_name);
        fs::write(&source, code).unwrap();
        let status = Command::new("rustc")
            .arg("--edition=2021")
            .arg(&source)
            .arg("-o")
            .arg(executable)
            .status()
            .expect("compile Windows test process");
        assert!(status.success(), "failed to compile Windows test process");
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

    #[cfg(unix)]
    fn test_script_command(path: &Path) -> Vec<String> {
        vec!["bash".to_string(), path.display().to_string()]
    }

    #[cfg(windows)]
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
}
