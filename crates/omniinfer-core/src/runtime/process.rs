use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
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
    #[error("runtime startup was interrupted")]
    Interrupted,
    #[error("runtime stop hook failed: {0}")]
    StopHook(String),
}

impl RuntimeProcess {
    pub fn start(
        plan: &ExternalRuntimePlan,
        options: RuntimeProcessOptions,
    ) -> Result<Self, RuntimeProcessError> {
        Self::start_with_cancellation(plan, options, None)
    }

    pub fn start_cancellable(
        plan: &ExternalRuntimePlan,
        options: RuntimeProcessOptions,
        cancelled: &AtomicBool,
    ) -> Result<Self, RuntimeProcessError> {
        Self::start_with_cancellation(plan, options, Some(cancelled))
    }

    fn start_with_cancellation(
        plan: &ExternalRuntimePlan,
        options: RuntimeProcessOptions,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self, RuntimeProcessError> {
        if is_cancelled(cancelled) {
            return Err(RuntimeProcessError::Interrupted);
        }
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
        if is_cancelled(cancelled) {
            let _ = terminate_runtime(
                &mut child,
                plan.stop_command.as_deref(),
                Duration::from_secs(2),
            );
            return Err(RuntimeProcessError::Interrupted);
        }
        let readiness = wait_runtime_ready(
            &options.health_host,
            plan.port,
            &plan.readiness_probe,
            options.startup_timeout,
            &mut child,
            &options.log_path,
            log_start_offset,
            cancelled,
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
    cancelled: Option<&AtomicBool>,
) -> Result<bool, RuntimeProcessError> {
    let deadline = Instant::now() + timeout;
    let mut log_cursor = log_start_offset;
    let mut log_tail = Vec::new();
    let mut log_marker_seen = false;
    while Instant::now() < deadline {
        if is_cancelled(cancelled) {
            return Err(RuntimeProcessError::Interrupted);
        }
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

fn is_cancelled(cancelled: Option<&AtomicBool>) -> bool {
    cancelled.is_some_and(|value| value.load(Ordering::SeqCst))
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
mod tests;
