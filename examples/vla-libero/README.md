# OmniInfer vla.cpp LIBERO live demo (Linux)

This example demonstrates the complete managed path:

```text
browser dashboard
  -> LIBERO simulator
  -> vla.cpp-compatible request preprocessing and protobuf client
  -> OmniInfer-managed vla-server
```

The dashboard shows the live front/wrist camera views, the 7-DoF action sent to
LIBERO, model/policy/simulator/control-loop latency, task text, episode progress,
and explicit success, failure, stop, or error results. LIBERO also writes the
episode MP4 under `--output-dir`.

The browser can select any of the ten predefined `libero_object` tasks before
starting a rollout. The task selector sends a LIBERO task id, not arbitrary
prompt text, so the language instruction, scene, target object, and success
condition remain consistent. Selection is locked while a rollout is active.
For multi-episode runs, the final status is `success`, `failed`, or `partial`;
`partial` means that the same run contained both successful and failed episodes.
It currently supports the SmolVLA and PI0.5 vla.cpp request formats.

This is an optional Linux developer example. It is not packaged with OmniInfer:
the setup process downloads LIBERO and creates its own Python environment, and
model files remain user-provided.

## Quick start

Run these commands from an OmniInfer checkout on Linux. They install only the
optional demo dependencies; no Python environment, LIBERO source, model, or
cache is added to the OmniInfer release package.

```sh
# 1. System tools (Ubuntu/Debian example).
sudo apt-get update
sudo apt-get install -y git protobuf-compiler

# 2. Install uv if it is not already available.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Fetch the vla.cpp submodule required by the demo client.
git submodule update --init framework/vla.cpp

# 4. Create the small CPU-only demo environment (default).
examples/vla-libero/setup.sh

# 5. Start OmniInfer, then start the dashboard in another terminal.
OMNIINFER_SERVE_DIRECT=1 ./omniinfer serve \
  --host 127.0.0.1 --port 9000 --no-restore-model

MUJOCO_GL=egl examples/vla-libero/run.sh -- \
  --backend vla.cpp-linux-cuda \
  --model <path-to-smolvla.gguf> \
  --arch smolvla --task-id 0 --episodes 1
```

Open `http://127.0.0.1:7860`, choose a task, and press **Start**. For a
remote Linux host, use `ssh -L 7860:127.0.0.1:7860 <host>` and open the same
local URL. The dashboard never exposes a network listener by default.

To create a Python environment with CUDA PyTorch instead of the CPU default:

```sh
examples/vla-libero/setup.sh --torch-backend cu124
```

Choose the CUDA option only when Python-side GPU preprocessing is required.
`vla-server` performs model inference independently, so the CPU environment is
the recommended default for this example.

## Simulation flow

The dashboard is deliberately a visible, end-to-end rollout rather than a
hidden benchmark runner:

1. Select one predefined `libero_object` task in the browser. Each choice fixes
   the language instruction, scene, object, and LIBERO success condition.
2. Press **Start**. The dashboard validates the selected task and asks
   OmniInfer to load the requested VLA model, or reuses the ready managed
   runtime when no `--model` is supplied.
3. OmniInfer launches or supervises `vla-server` and reports its loopback
   ZeroMQ/protobuf endpoint. The dashboard rejects any non-vla.cpp or
   non-loopback endpoint.
4. LIBERO/MuJoCo resets the selected scene and yields front-camera, wrist-camera
   and robot-state observations.
5. The demo converts those observations to the vla.cpp request format and sends
   them directly to the OmniInfer-managed `vla-server`.
6. `vla-server` returns an action chunk. The demo applies the next 7-DoF action
   to LIBERO, then repeats the observation → action loop until the episode ends
   or the user presses **Stop**.
7. Every step updates the browser with the live camera frame, current action,
   task text, episode/step count, and model/policy/simulator/control-loop
   latency. Each rollout writes an MP4 to its own timestamped subdirectory
   under `--output-dir`.
8. LIBERO reports the final success condition. The page explicitly displays
   `success`, `failed`, `partial`, `stopped`, or `error`; it never treats a
   completed process as success by itself.

## Prerequisites

1. Linux x86_64, Python 3.10, Git, `uv`, and `protoc`.
2. An NVIDIA driver plus EGL-capable rendering for CUDA vla.cpp runtimes.
3. Build OmniInfer and install/build `vla.cpp-linux` or
   `vla.cpp-linux-cuda` from this checkout.
4. Initialize `framework/vla.cpp`.
5. Have a vla.cpp-compatible checkpoint and the tokenizer/stats required by its
   architecture.

The vla.cpp Python client invokes `protoc` to generate its local protobuf stub.
If `protoc` is installed outside `PATH`, pass `--protoc <path-to-protoc>`; the
demo validates it before initializing the client.

When the tokenizer is already cached (or `--tokenizer` points to a local
snapshot) and the demonstration host has no Internet access, set
`HF_HUB_OFFLINE=1` to avoid Hugging Face Hub connection retries during startup.

Start a loopback OmniInfer gateway in direct mode:

```sh
OMNIINFER_SERVE_DIRECT=1 ./omniinfer serve \
  --host 127.0.0.1 \
  --port 9000 \
  --no-restore-model
```

Create the dedicated dashboard environment:

```sh
examples/vla-libero/setup.sh
```

This clones the upstream LIBERO source checkout on first use and creates an
isolated environment under `${XDG_CACHE_HOME:-~/.cache}/omniinfer/`. It does
not install or modify vla.cpp's complete `setup_libero.sh` evaluation
environment. The dashboard's default is CPU-only PyTorch: model inference
continues to run in the separately managed `vla-server`, so Python needs torch
only for simulation and request preprocessing. On the validation host this
reduced the environment from about 9.6 GB to about 2.3 GB.

To retain a CUDA PyTorch environment, request one explicitly:

```sh
examples/vla-libero/setup.sh --torch-backend cu124
```

The CUDA option is for compatibility or local GPU-side preprocessing; it does
not move vla-server inference into Python. `uv` and a system `protoc` are still
required. `setup.sh --help` documents alternate venv, LIBERO source, and uv
paths.

## SmolVLA

```sh
MUJOCO_GL=egl examples/vla-libero/run.sh -- \
  --backend vla.cpp-linux-cuda \
  --model <path-to-smolvla.gguf> \
  --arch smolvla \
  --task libero_object \
  --task-id 0 \
  --episodes 1 \
  --n-action-steps 1
```

## PI0.5

```sh
MUJOCO_GL=egl examples/vla-libero/run.sh -- \
  --backend vla.cpp-linux-cuda \
  --model <path-to-pi05.gguf> \
  --arch pi05 \
  --tokenizer <path-or-id-to-paligemma-tokenizer> \
  --stats-json <path-to-libero-meta-stats.json> \
  --task libero_object \
  --task-id 0 \
  --episodes 1 \
  --n-action-steps 10
```

If OmniInfer already manages the desired VLA model, omit `--model`; the demo
validates `/omni/state` and uses its reported `client_endpoint`. It rejects a
non-VLA protocol, non-VLA backend, missing endpoint, and non-loopback ZMQ
endpoint instead of silently connecting to the wrong runtime.

If the gateway uses an admin key, place it in
`OMNIINFER_ADMIN_API_KEY`; the demo reads the environment variable and sends a
Bearer header without putting the secret in the process command line.

Open `http://127.0.0.1:7860`. The dashboard is intentionally loopback-only:
it can start and stop GPU rollouts, so it must not be exposed directly to a
network. When it runs on a remote machine, forward it over SSH:

```sh
ssh -L 7860:127.0.0.1:7860 <remote-host>
```

The page is idle on startup. Choose a predefined task and press **Start**;
arbitrary text is deliberately not accepted, because each LIBERO task binds its
instruction, scene, object, and success condition. Stop requests take effect
after the current policy or simulator step returns.

## Files and cleanup

- `setup.sh` downloads LIBERO under `framework/vla.cpp/eval/sim/libero/LIBERO`
  and creates a venv under `${XDG_CACHE_HOME:-~/.cache}/omniinfer/` by default.
- `run.sh` only starts the dashboard; it never installs packages.
- `demo.py` never installs Python packages. A tokenizer may still be fetched by
  `transformers` if `--tokenizer` is not local/cached; set `HF_HUB_OFFLINE=1`
  to require offline use.
- Remove the chosen venv and LIBERO checkout manually when no longer needed.
  Neither belongs in Git or the OmniInfer release archive.

## Metric semantics

- **Model prediction**: preprocessing plus the synchronous vla.cpp
  request/response on steps that request a new action chunk.
- **Policy call**: every action request, including inexpensive action-queue
  replay when `--n-action-steps` is greater than one.
- **Simulator step**: the LIBERO environment step.
- **Control loop**: policy call plus simulator step.

Latency cards show P50 as the primary value and also report mean and P95 over
the most recent 500 steps. This keeps an episode-ending LIBERO environment
reset visible in the aggregate without presenting that terminal reset as the
normal per-step latency.

The dashboard is a rollout demonstration, not a full LIBERO benchmark. Use
vla.cpp's official evaluation runners for benchmark-scale success-rate claims.
