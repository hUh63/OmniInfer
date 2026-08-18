use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process;
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    record_invocation(&args);
    if args
        .first()
        .is_some_and(|arg| arg == "--query-gpu=index,uuid,memory.free")
    {
        println!("0, GPU-fake, 65536");
        return;
    }
    if args
        .first()
        .is_some_and(|arg| arg == "--query-gpu=index")
    {
        println!("0");
        return;
    }
    if args.first().is_some_and(|arg| arg.starts_with("--query-gpu")) {
        println!("581.57");
        return;
    }
    if args.as_slice() == ["--list", "--quiet"] {
        println!("Ubuntu-24.04");
        return;
    }
    if args.as_slice() == ["--list", "--verbose"] {
        println!("  NAME STATE VERSION");
        println!("* Ubuntu-24.04 Running 2");
        return;
    }
    let Some(exec_index) = args.iter().position(|arg| arg == "--exec") else {
        fail("fake WSL expected --exec");
    };
    let command = &args[exec_index + 1..];
    if command.is_empty() {
        fail("fake WSL command is empty");
    }
    match command[0].as_str() {
        "sh" => handle_sh(command),
        "wslpath" => handle_wslpath(command),
        "uname" => println!("x86_64"),
        "nvidia-smi" => println!("NVIDIA GeForce RTX 3060 Laptop GPU, 581.57"),
        "env" => handle_env(command),
        "install" => handle_install(command),
        "apt-get" | "dpkg" | "/sbin/ldconfig" => {}
        "dpkg-query" => {
            if !root().join("rocm-installed").exists() {
                return;
            }
            println!("comgr=3.0.0.70203-90~24.04");
            println!("hipblas=3.2.0.70203-90~24.04");
            println!("hipblaslt=1.2.2.70203-90~24.04");
            println!("hipfft=1.0.22.70203-90~24.04");
            println!("hiprand=3.1.0.70203-90~24.04");
            println!("hip-runtime-amd=7.2.53211.70203-90~24.04");
            println!("hipsolver=3.2.0.70203-90~24.04");
            println!("hipsparse=4.2.0.70203-90~24.04");
            println!("hipsparselt=0.2.6.70203-90~24.04");
            println!("hsa-rocr=1.18.0.70203-90~24.04");
            println!("libopenmpi3t64=4.1.6-7ubuntu2");
            if root().join("python-dev-installed").exists() {
                println!("libpython3.12-dev=3.12.3-1ubuntu0.15");
            }
            println!("miopen-hip=3.5.1.70203-90~24.04");
            println!("openmp-extras-runtime=20.70.0.70203-90~24.04");
            if root().join("python-dev-installed").exists() {
                println!("python3.12-dev=3.12.3-1ubuntu0.15");
            }
            println!("rccl=2.27.7.70203-90~24.04");
            println!("rocblas=5.2.0.70203-90~24.04");
            println!("rocfft=1.0.36.70203-90~24.04");
            println!("rocm-hip-runtime=7.2.3.70203-90~24.04");
            println!("rocm-core=7.2.3.70203-90~24.04");
            println!("rocm-device-libs=1.0.0.70203-90~24.04");
            println!("rocm-language-runtime=7.2.3.70203-90~24.04");
            println!("rocm-llvm=22.0.0.26084.70203-90~24.04");
            println!("rocm-smi-lib=7.8.0.70203-90~24.04");
            println!("rocminfo=1.0.0.70203-90~24.04");
            println!("rocprofiler-register=0.6.0.70203-90~24.04");
            println!("rocrand=4.2.0.70203-90~24.04");
            println!("rocsolver=3.32.0.70203-90~24.04");
            println!("rocsparse=4.2.0.70203-90~24.04");
            println!("roctracer=4.1.70203.70203-90~24.04");
            println!("rocdxg-roct=1.2.0");
        }
        "sha256sum" => {
            let path = command.get(1).expect("sha256sum path");
            let mapped = map_linux(path);
            if !mapped.exists() {
                fail(&format!("sha256sum: {path}: No such file"));
            }
            let sidecar = mapped.with_extension("sha256");
            let sidecar_digest = fs::read_to_string(&sidecar).ok();
            let digest = sidecar_digest.as_deref().unwrap_or_else(|| {
                Path::new(path)
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.rsplit('-').next())
                    .expect("digest in staged filename")
            });
            println!("{digest}  {path}");
        }
        "mkdir" => handle_mkdir(command),
        "dd" => handle_dd(command),
        "rm" => handle_rm(command),
        "cp" => handle_cp(command),
        "chmod" => {}
        "tee" => handle_tee(command),
        executable if executable.ends_with("/omniinfer-vllm-run") => handle_runner(command),
        executable if executable.ends_with("/omniinfer-vllm-stop") => handle_stopper(command),
        executable if executable.ends_with("/uv-0.11.16") => handle_uv(command),
        executable if executable.ends_with("/venv/bin/python") => {
            if env::var_os("OMNIINFER_FAKE_WSL_FAIL_CURRENT_PROBE").is_some()
                && executable.contains("/current/")
            {
                fail("injected current runtime probe failure");
            }
            println!(
                r#"{{"vllm_version":"0.24.0+cu129","torch_version":"2.11.0+cu129","torch_cuda":"12.9","device":"Fake CUDA","value":1.0}}"#
            );
        }
        other => fail(&format!("unsupported fake WSL command: {other}")),
    }
}

fn handle_sh(command: &[String]) {
    let script = command.get(2).map(String::as_str).unwrap_or_default();
    if script.contains("printf %s \"$HOME\"") {
        print!("/home/test");
        return;
    }
    if script.contains("/etc/os-release") {
        print!("ubuntu 24.04");
        return;
    }
    if script.contains("test -d \"$staging\"") {
        let staging = map_linux(command.get(5).expect("staging argument"));
        let current = map_linux(command.get(6).expect("current argument"));
        let backup = map_linux(command.get(7).expect("backup argument"));
        remove(&backup);
        if current.exists() {
            rename(&current, &backup);
        }
        rename(&staging, &current);
        return;
    }
    if script.contains("rm -rf \"$current\"") {
        let current = map_linux(command.get(4).expect("current argument"));
        let backup = map_linux(command.get(5).expect("backup argument"));
        remove(&current);
        if backup.exists() {
            rename(&backup, &current);
        }
        return;
    }
    if script.contains("for pid_file in") {
        return;
    }
    if script.contains("managed native extensions") {
        println!("42");
        return;
    }
    if script.contains("exec \"$python\" -c \"$probe\"") {
        let runtime = command.get(4).map(String::as_str).unwrap_or_default();
        if env::var_os("OMNIINFER_FAKE_WSL_FAIL_CURRENT_PROBE").is_some()
            && runtime.contains("/current")
        {
            fail("injected current runtime probe failure");
        }
        if runtime.contains("vllm-wsl2-rocm") {
            println!(
                r#"{{"vllm_version":"0.26.0","torch_version":"2.11.0+rocm7.2.3","torch_cuda":null,"torch_hip":"7.2.53211","device":"Fake AMD Radeon 8060S","value":1.0}}"#
            );
        } else {
            println!(
                r#"{{"vllm_version":"0.24.0+cu129","torch_version":"2.11.0+cu129","torch_cuda":"12.9","device":"Fake CUDA","value":1.0}}"#
            );
        }
        return;
    }
    fail("unsupported fake WSL shell script");
}

fn handle_env(command: &[String]) {
    if command.iter().any(|arg| arg == "apt-get") {
        fs::create_dir_all(root()).expect("create fake WSL root");
        fs::write(root().join("rocm-installed"), b"installed")
            .expect("mark fake ROCm packages installed");
        if command
            .iter()
            .any(|arg| arg == "python3.12-dev=3.12.3-1ubuntu0.15")
            && command
                .iter()
                .any(|arg| arg == "libpython3.12-dev=3.12.3-1ubuntu0.15")
        {
            fs::write(root().join("python-dev-installed"), b"installed")
                .expect("mark fake Python development packages installed");
        }
        return;
    }
    if command
        .iter()
        .any(|arg| arg == "/opt/rocm/bin/rocminfo")
    {
        println!("  Name:                    gfx1151");
        return;
    }
    if command
        .iter()
        .any(|arg| arg.ends_with("/venv/bin/python"))
    {
        if env::var_os("OMNIINFER_FAKE_WSL_FAIL_CURRENT_PROBE").is_some()
            && command.iter().any(|arg| arg.contains("/current/"))
        {
            fail("injected current runtime probe failure");
        }
        println!(
            r#"{{"vllm_version":"0.26.0","torch_version":"2.11.0+rocm7.2.3","torch_cuda":null,"torch_hip":"7.2.53211","device":"Fake AMD Radeon 8060S","value":1.0}}"#
        );
        return;
    }
    fail("unsupported fake WSL env command");
}

fn handle_install(command: &[String]) {
    if command.iter().any(|arg| arg == "-d") {
        for path in command.iter().skip(1).filter(|arg| {
            !arg.starts_with('-')
                && arg.as_str() != "0755"
                && arg.as_str() != "0644"
        }) {
            fs::create_dir_all(map_linux(path)).expect("create fake install directory");
        }
        return;
    }
    if command.len() < 3 {
        return;
    }
    let source = command.get(command.len() - 2).expect("install source");
    let destination = command.last().expect("install destination");
    let source_path = map_linux(source);
    if !source_path.exists() {
        return;
    }
    let destination_path = map_linux(destination);
    if let Some(parent) = destination_path.parent() {
        fs::create_dir_all(parent).expect("create fake install parent");
    }
    fs::copy(&source_path, &destination_path).expect("copy fake installed file");
    if let Some(digest) = Path::new(source)
        .file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| name.len() == 64 && name.chars().all(|ch| ch.is_ascii_hexdigit()))
    {
        fs::write(destination_path.with_extension("sha256"), digest)
            .expect("write fake installed checksum");
    }
}

fn handle_wslpath(command: &[String]) {
    let input = command.last().expect("wslpath input").replace('\\', "/");
    if input.eq_ignore_ascii_case("C:/") {
        print!("/mnt/c/");
        return;
    }
    let bytes = input.as_bytes();
    if bytes.len() >= 3 && bytes[1] == b':' {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        print!("/mnt/{drive}/{}", input[3..].trim_start_matches('/'));
        return;
    }
    print!("{input}");
}

fn handle_mkdir(command: &[String]) {
    for path in command.iter().skip(1).filter(|arg| *arg != "-p") {
        fs::create_dir_all(map_linux(path)).expect("create fake WSL directory");
    }
}

fn handle_dd(command: &[String]) {
    let output = command
        .iter()
        .find_map(|arg| arg.strip_prefix("of="))
        .expect("dd output path");
    let path = map_linux(output);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create dd parent");
    }
    let mut bytes = Vec::new();
    io::stdin().read_to_end(&mut bytes).expect("read dd stdin");
    fs::write(path, bytes).expect("write fake WSL dd output");
}

fn handle_rm(command: &[String]) {
    for path in command.iter().skip(1).filter(|arg| !arg.starts_with('-')) {
        remove(&map_linux(path));
    }
}

fn handle_cp(command: &[String]) {
    let destination = map_linux(command.last().expect("copy destination"));
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).expect("create copy parent");
    }
    fs::write(destination, b"fake uv").expect("write copied fake asset");
}

fn handle_tee(command: &[String]) {
    let path = map_linux(command.get(1).expect("tee path"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create tee parent");
    }
    let mut bytes = Vec::new();
    io::stdin().read_to_end(&mut bytes).expect("read tee stdin");
    fs::write(path, bytes).expect("write fake WSL file");
}

fn handle_uv(command: &[String]) {
    if command.get(1).map(String::as_str) == Some("venv") {
        let runtime = map_linux(command.last().expect("venv path"));
        let bin = runtime.join("bin");
        fs::create_dir_all(&bin).expect("create fake venv");
        fs::write(bin.join("python"), b"fake python").expect("write fake python");
        fs::write(bin.join("vllm"), b"fake vllm").expect("write fake vllm");
    }
}

fn handle_runner(command: &[String]) {
    let pid_file = map_linux(command.get(1).expect("runner pid file"));
    let stop_file = pid_file.with_extension("stop");
    remove(&stop_file);
    if let Some(parent) = pid_file.parent() {
        fs::create_dir_all(parent).expect("create fake runner pid parent");
    }
    fs::write(&pid_file, process::id().to_string()).expect("write fake runner pid");
    let port = command
        .windows(2)
        .find(|args| args[0] == "--port")
        .map(|args| args[1].parse::<u16>().expect("valid fake vLLM port"))
        .expect("fake vLLM --port");
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind fake vLLM server");
    listener
        .set_nonblocking(true)
        .expect("set fake vLLM nonblocking");
    while !stop_file.exists() {
        match listener.accept() {
            Ok((stream, _)) => handle_http(stream),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => fail(&format!("fake vLLM accept failed: {error}")),
        }
    }
    remove(&pid_file);
    remove(&stop_file);
}

fn handle_stopper(command: &[String]) {
    let pid_file = map_linux(command.get(1).expect("stopper pid file"));
    if !pid_file.exists() {
        return;
    }
    fs::write(pid_file.with_extension("stop"), b"stop").expect("write fake runner stop marker");
    let deadline = Instant::now() + Duration::from_secs(5);
    while pid_file.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    if pid_file.exists() {
        fail("fake vLLM runner did not stop");
    }
}

fn handle_http(mut stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone fake vLLM stream"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut content_length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.is_empty() {
            return;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }
    let response = if request_line.starts_with("GET /health") {
        r#"{"status":"ok"}"#
    } else if request_line.starts_with("POST /v1/chat/completions") {
        r#"{"choices":[{"message":{"content":"fake vLLM WSL2"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#
    } else {
        r#"{"ok":true}"#
    };
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.len()
    );
    let _ = stream.write_all(headers.as_bytes());
    let _ = stream.write_all(response.as_bytes());
}

fn record_invocation(args: &[String]) {
    fs::create_dir_all(root()).expect("create fake WSL root");
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root().join("invocations.log"))
        .expect("open fake WSL invocation log");
    writeln!(log, "{}", args.join("\t")).expect("record fake WSL invocation");
}

fn root() -> PathBuf {
    PathBuf::from(env::var_os("OMNIINFER_FAKE_WSL_ROOT").expect("fake WSL root"))
}

fn map_linux(path: impl AsRef<str>) -> PathBuf {
    let path = path.as_ref().trim_start_matches('/');
    root().join("linux").join(path.replace('/', "\\"))
}

fn remove(path: &Path) {
    if path.is_dir() {
        fs::remove_dir_all(path).ok();
    } else {
        fs::remove_file(path).ok();
    }
}

fn rename(source: &Path, target: &Path) {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).expect("create rename parent");
    }
    fs::rename(source, target).expect("rename fake WSL path");
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    process::exit(2);
}
