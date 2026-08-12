#!/usr/bin/env python3
"""Live OmniInfer + vla.cpp + LIBERO demonstration dashboard."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


VLA_PROTOCOL = "vla.cpp-zmq-server"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
LIBERO_OBJECT_TASKS = (
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
)
SUPPORTED_DEMO_ARCHES = ("smolvla", "pi05")


def validate_libero_object_task_id(value: Any) -> int:
    """Return a task id only when it identifies a supported LIBERO object task."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("task_id must be an integer")
    if not 0 <= value < len(LIBERO_OBJECT_TASKS):
        raise ValueError(
            f"task_id must be between 0 and {len(LIBERO_OBJECT_TASKS) - 1}"
        )
    return value


def aggregate_result(successes: int, failures: int) -> str:
    """Describe a completed multi-episode rollout without hiding mixed results."""
    if successes and failures:
        return "partial"
    if successes:
        return "success"
    return "failed"


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for a finite sample."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "last_ms": round(values[-1], 2) if values else None,
        "mean_ms": round(statistics.fmean(values), 2) if values else None,
        "p50_ms": round(percentile(values, 0.50), 2) if values else None,
        "p95_ms": round(percentile(values, 0.95), 2) if values else None,
    }


def validate_vla_runtime(payload: dict[str, Any]) -> tuple[str, str, str | None]:
    """Validate OmniInfer's managed VLA protocol response and return its endpoint."""
    protocol = payload.get("external_server_protocol")
    endpoint = payload.get("client_endpoint")
    backend = payload.get("selected_backend") or payload.get("backend")
    model = payload.get("selected_model") or payload.get("model_path") or payload.get("model")

    if protocol != VLA_PROTOCOL:
        raise ValueError(
            f"OmniInfer runtime protocol is {protocol!r}; expected {VLA_PROTOCOL!r}."
        )
    if not isinstance(backend, str) or not backend.startswith("vla.cpp-"):
        raise ValueError(f"OmniInfer selected a non-VLA backend: {backend!r}.")
    if not isinstance(endpoint, str):
        raise ValueError("OmniInfer did not report a VLA client_endpoint.")

    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "tcp" or parsed.hostname not in LOOPBACK_HOSTS or parsed.port is None:
        raise ValueError(
            "VLA client_endpoint must be a loopback tcp:// endpoint reported by OmniInfer; "
            f"got {endpoint!r}."
        )
    return endpoint, backend, str(model) if model is not None else None


@dataclass(frozen=True)
class DemoConfig:
    omniinfer_url: str = "http://127.0.0.1:9000"
    backend: str = "vla.cpp-linux-cuda"
    model: str | None = None
    mmproj: str | None = None
    launch_args: tuple[str, ...] = ()
    admin_api_key: str | None = None
    protoc: str | None = None
    arch: str = "smolvla"
    tokenizer: str | None = None
    stats_json: str | None = None
    unnorm_key: str | None = None
    task: str = "libero_object"
    task_id: int = 0
    episodes: int = 1
    seed: int = 42
    fps: int = 20
    output_dir: str = "outputs/vla-libero"
    view_mode: str = "multi-view"
    n_action_steps: int = 1
    recv_timeout_ms: int = 120_000

    def model_load_payload(self) -> dict[str, Any] | None:
        if self.model is None:
            return None
        payload: dict[str, Any] = {
            "model": self.model,
            "backend": self.backend,
            "strict_capabilities": True,
        }
        if self.mmproj:
            payload["mmproj"] = self.mmproj
        if self.launch_args:
            payload["launch_args"] = list(self.launch_args)
        return payload


class OmniInferAPI:
    def __init__(self, base_url: str, admin_api_key: str | None = None):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid OmniInfer URL: {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.admin_api_key = admin_api_key

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.admin_api_key:
            headers["Authorization"] = f"Bearer {self.admin_api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=450) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OmniInfer {path} returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Cannot reach OmniInfer at {self.base_url}: {error}") from error

    def resolve_vla_runtime(self, config: DemoConfig) -> tuple[str, str, str | None]:
        load_payload = config.model_load_payload()
        if load_payload is not None:
            payload = self._request("/omni/model/load", load_payload)
        else:
            payload = self._request("/omni/state")
            if not payload.get("backend_ready"):
                raise RuntimeError(
                    "No managed runtime is ready. Pass --model or load a vla.cpp backend first."
                )
        return validate_vla_runtime(payload)


def configure_protoc(protoc: str | None) -> str:
    """Make the protoc required by vla.cpp's Python client explicit and discoverable."""
    if protoc:
        candidate = Path(protoc).expanduser().resolve()
        if not candidate.is_file() or shutil.which(str(candidate)) is None:
            raise ValueError(f"--protoc is not an executable file: {candidate}")
        os.environ["PATH"] = f"{candidate.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    resolved = shutil.which("protoc")
    if resolved is None:
        raise RuntimeError(
            "vla.cpp's Python client requires protoc to generate its protobuf stub. "
            "Install protobuf-compiler or pass --protoc <path>."
        )
    return resolved


class DemoState:
    def __init__(self, config: DemoConfig):
        self._lock = threading.Lock()
        self._frame = b""
        self._policy_ms: deque[float] = deque(maxlen=500)
        self._prediction_ms: deque[float] = deque(maxlen=500)
        self._env_ms: deque[float] = deque(maxlen=500)
        self._loop_ms: deque[float] = deque(maxlen=500)
        self._events: deque[dict[str, Any]] = deque(maxlen=80)
        self._data: dict[str, Any] = {
            "phase": "idle",
            "result": "idle",
            "message": "Ready to start",
            "task": config.task,
            "task_id": config.task_id,
            "task_description": LIBERO_OBJECT_TASKS[config.task_id],
            "task_options": [
                {"task_id": task_id, "instruction": instruction}
                for task_id, instruction in enumerate(LIBERO_OBJECT_TASKS)
            ],
            "arch": config.arch,
            "backend": config.backend,
            "model": config.model,
            "client_endpoint": None,
            "episode": 0,
            "episodes": config.episodes,
            "step": 0,
            "successes": 0,
            "failures": 0,
            "reward": 0.0,
            "action": [0.0] * len(ACTION_LABELS),
            "action_labels": ACTION_LABELS,
            "call_kind": None,
            "frame_seq": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    def begin(self, task_id: int) -> None:
        with self._lock:
            self._frame = b""
            self._policy_ms.clear()
            self._prediction_ms.clear()
            self._env_ms.clear()
            self._loop_ms.clear()
            self._events.clear()
            self._data.update(
                phase="starting",
                result="running",
                message="Starting OmniInfer-managed VLA runtime",
                task_id=task_id,
                task_description=LIBERO_OBJECT_TASKS[task_id],
                client_endpoint=None,
                episode=0,
                step=0,
                successes=0,
                failures=0,
                reward=0.0,
                action=[0.0] * len(ACTION_LABELS),
                call_kind=None,
                frame_seq=0,
                started_at=time.time(),
                finished_at=None,
                error=None,
            )
            self._append_event_locked("info", "Demo run started")

    def update(self, **values: Any) -> None:
        with self._lock:
            self._data.update(values)

    def event(self, level: str, message: str) -> None:
        with self._lock:
            self._append_event_locked(level, message)

    def _append_event_locked(self, level: str, message: str) -> None:
        self._events.append({"time": time.time(), "level": level, "message": message})

    def publish_frame(self, frame: bytes) -> None:
        with self._lock:
            self._frame = frame
            self._data["frame_seq"] += 1

    def publish_step(
        self,
        *,
        action: list[float],
        policy_ms: float,
        prediction_sent: bool,
        env_ms: float,
        loop_ms: float,
        reward: float,
        step: int,
    ) -> None:
        with self._lock:
            self._policy_ms.append(policy_ms)
            if prediction_sent:
                self._prediction_ms.append(policy_ms)
            self._env_ms.append(env_ms)
            self._loop_ms.append(loop_ms)
            self._data.update(
                action=[round(float(value), 5) for value in action],
                call_kind="model_prediction" if prediction_sent else "action_queue_replay",
                reward=round(float(reward), 5),
                step=step,
                message="Running LIBERO rollout",
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._data)
            payload["events"] = list(self._events)
            payload["latency"] = {
                "policy": metric_summary(list(self._policy_ms)),
                "prediction": metric_summary(list(self._prediction_ms)),
                "environment": metric_summary(list(self._env_ms)),
                "control_loop": metric_summary(list(self._loop_ms)),
                "history_ms": [round(value, 2) for value in list(self._loop_ms)[-80:]],
            }
            return payload

    def frame(self) -> bytes:
        with self._lock:
            return self._frame


def encode_frame(observation: dict[str, Any], view_mode: str) -> bytes:
    import numpy as np
    from PIL import Image

    front = np.asarray(observation["pixels"]["image"][::-1, ::-1], dtype=np.uint8)
    images = [front]
    if view_mode == "multi-view" and "image2" in observation["pixels"]:
        wrist = np.asarray(observation["pixels"]["image2"][::-1, ::-1], dtype=np.uint8)
        if wrist.shape[0] != front.shape[0]:
            wrist = np.asarray(
                Image.fromarray(wrist).resize(
                    (round(wrist.shape[1] * front.shape[0] / wrist.shape[0]), front.shape[0])
                )
            )
        images.append(wrist)
    composed = np.concatenate(images, axis=1)
    output = io.BytesIO()
    Image.fromarray(composed).save(output, format="JPEG", quality=84, optimize=False)
    return output.getvalue()


class _DemoPolicyAdapter:
    """Keep demo-side LIBERO conversion independent from LeRobot's full package."""

    def __init__(self, client: Any):
        self._client = client

    def reset(self) -> None:
        self._client.reset()

    def get_action(self, observation: dict[str, Any]) -> Any:
        return self._client.get_action(self.parse_observation(observation))

    def parse_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _quat_to_axis_angle(quat: Any) -> Any:
        import numpy as np

        quat = np.asarray(quat, dtype=np.float32).reshape(4)
        w = float(np.clip(quat[3], -1.0, 1.0))
        denominator = float(np.sqrt(max(0.0, 1.0 - w * w)))
        if denominator <= 1e-10:
            return np.zeros(3, dtype=np.float32)
        return (quat[:3] * (2.0 * np.arccos(w) / denominator)).astype(np.float32)


class _LiberoPolicyAdapter(_DemoPolicyAdapter):
    """Equivalent LIBERO image/state conversion for vla.cpp's direct client."""

    def parse_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        images = observation["pixels"]

        def image(key: str) -> Any:
            # vla.cpp's official LiberoProcessorStep flips both axes after
            # converting HWC uint8 to CHW float32 in [0, 1].
            value = np.asarray(images[key], dtype=np.float32)
            value = value[::-1, ::-1].transpose(2, 0, 1) / 255.0
            return np.ascontiguousarray(value, dtype=np.float32)

        robot_state = observation["robot_state"]
        state = np.concatenate(
            (
                np.asarray(robot_state["eef"]["pos"], dtype=np.float32),
                self._quat_to_axis_angle(robot_state["eef"]["quat"]),
                np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32),
            )
        )
        return {
            "observation.images.image": image("image"),
            "observation.images.image2": image("image2"),
            "observation.state": np.ascontiguousarray(state, dtype=np.float32),
            "task": observation.get("task_description", ""),
        }


def create_policy(config: DemoConfig, endpoint: str) -> tuple[Any, Any]:
    from client.vla_cpp_client import VlaCppClient

    raw_client = VlaCppClient(
        vla_addr=endpoint,
        arch=config.arch,
        tokenizer_name=config.tokenizer,
        recv_timeout_ms=config.recv_timeout_ms,
        n_action_steps=config.n_action_steps,
        stats_json=config.stats_json,
        bitvla_unnorm_key=config.unnorm_key,
    )
    return raw_client, _LiberoPolicyAdapter(client=raw_client)


class DemoController:
    def __init__(self, config: DemoConfig, state: DemoState, repository_root: Path):
        self.config = config
        self.state = state
        self.repository_root = repository_root
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, task_id: int | None = None) -> bool:
        with self._guard:
            if self._thread is not None and self._thread.is_alive():
                return False
            if task_id is None:
                task_id = int(self.state.snapshot()["task_id"])
            task_id = validate_libero_object_task_id(task_id)
            self._stop.clear()
            self.state.begin(task_id)
            self._thread = threading.Thread(
                target=self._run_guarded, args=(task_id,), daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> bool:
        with self._guard:
            running = self._thread is not None and self._thread.is_alive()
            if running:
                self._stop.set()
                self.state.update(message="Stopping after the current simulator step")
            return running

    def _run_guarded(self, task_id: int) -> None:
        try:
            self._run(task_id)
        except Exception as error:  # dashboard must preserve the failure for inspection
            self.state.event("error", str(error))
            self.state.update(
                phase="error",
                result="error",
                message="Demo failed",
                error=f"{type(error).__name__}: {error}",
                finished_at=time.time(),
            )
            traceback.print_exc()
        finally:
            with self._guard:
                self._thread = None

    def _run(self, task_id: int) -> None:
        eval_root = self.repository_root / "framework" / "vla.cpp" / "eval"
        if not (eval_root / "client" / "vla_cpp_client.py").is_file():
            raise RuntimeError(
                "framework/vla.cpp is not initialized; run git submodule update --init framework/vla.cpp"
            )
        sys.path.insert(0, str(eval_root))

        api = OmniInferAPI(self.config.omniinfer_url, self.config.admin_api_key)
        endpoint, backend, model = api.resolve_vla_runtime(self.config)
        self.state.update(
            phase="initializing",
            message="Initializing tokenizer and LIBERO simulator",
            backend=backend,
            model=model,
            client_endpoint=endpoint,
        )
        self.state.event("info", f"OmniInfer runtime ready: {backend} at {endpoint}")

        import gymnasium as gym
        import sim.libero  # noqa: F401 - registers the LIBERO environments

        raw_client = None
        environment = None
        try:
            protoc = configure_protoc(self.config.protoc)
            self.state.event("info", f"Using protoc: {protoc}")
            raw_client, policy = create_policy(self.config, endpoint)
            output_dir = Path(self.config.output_dir).resolve()
            run_dir = output_dir / (
                f"run-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}-task-{task_id}"
            )
            run_dir.mkdir(parents=True, exist_ok=False)
            self.state.event("info", f"Writing rollout video to {run_dir}")
            environment = gym.make(
                f"{self.config.task}/task_{task_id}",
                seed=self.config.seed,
                video_fps=self.config.fps,
                output_video_dir=run_dir,
                video_view_mode=self.config.view_mode,
            )
            self.state.update(phase="running", message="Running LIBERO rollout")

            successes = 0
            failures = 0
            for episode_index in range(self.config.episodes):
                if self._stop.is_set():
                    break
                policy.reset()
                observation, _ = environment.reset()
                self.state.publish_frame(encode_frame(observation, self.config.view_mode))
                self.state.update(
                    phase="running",
                    result="running",
                    message="Running LIBERO rollout",
                    episode=episode_index + 1,
                    step=0,
                    task_description=observation.get("task_description", ""),
                    reward=0.0,
                    error=None,
                )
                self.state.event(
                    "info", f"Episode {episode_index + 1}/{self.config.episodes} started"
                )

                actions_until_prediction = 0
                episode_finished = False
                step = 0
                while not self._stop.is_set():
                    prediction_sent = actions_until_prediction == 0
                    loop_start = time.perf_counter()
                    policy_start = loop_start
                    action = policy.get_action(observation)
                    policy_ms = (time.perf_counter() - policy_start) * 1000.0
                    if prediction_sent:
                        actions_until_prediction = self.config.n_action_steps - 1
                    else:
                        actions_until_prediction -= 1

                    env_start = time.perf_counter()
                    try:
                        observation, reward, terminated, truncated, info = environment.step(action)
                    except ValueError as error:
                        if "terminated episode" not in str(error):
                            raise
                        failures += 1
                        self.state.event("error", f"Episode aborted by LIBERO: {error}")
                        self.state.update(
                            result="failed",
                            message="LIBERO aborted the episode",
                            failures=failures,
                            error=str(error),
                        )
                        episode_finished = True
                        break
                    env_ms = (time.perf_counter() - env_start) * 1000.0
                    loop_ms = (time.perf_counter() - loop_start) * 1000.0
                    step += 1
                    self.state.publish_frame(encode_frame(observation, self.config.view_mode))
                    self.state.publish_step(
                        action=list(action[: len(ACTION_LABELS)]),
                        policy_ms=policy_ms,
                        prediction_sent=prediction_sent,
                        env_ms=env_ms,
                        loop_ms=loop_ms,
                        reward=reward,
                        step=step,
                    )

                    if terminated or truncated:
                        success = bool(info.get("is_success", False))
                        if success:
                            successes += 1
                            result = "success"
                            message = "Task completed successfully"
                        else:
                            failures += 1
                            result = "failed"
                            message = "Episode ended without task success"
                        self.state.update(
                            result=result,
                            message=message,
                            successes=successes,
                            failures=failures,
                        )
                        self.state.event(
                            "success" if success else "warning",
                            f"Episode {episode_index + 1}: {result} after {step} steps",
                        )
                        episode_finished = True
                        break

                if self._stop.is_set():
                    break
                if not episode_finished:
                    failures += 1
                    self.state.update(failures=failures, result="failed")

            if self._stop.is_set():
                self.state.update(
                    phase="stopped",
                    result="stopped",
                    message="Demo stopped by user",
                    finished_at=time.time(),
                )
                self.state.event("warning", "Demo stopped by user")
            else:
                result = aggregate_result(successes, failures)
                self.state.update(
                    phase="completed",
                    result=result,
                    message=(
                        f"Completed {self.config.episodes} episode(s): "
                        f"{successes} success, {failures} failure"
                    ),
                    successes=successes,
                    failures=failures,
                    finished_at=time.time(),
                )
                self.state.event(
                    "info", f"Run complete: {successes} success, {failures} failure"
                )
        finally:
            if environment is not None:
                environment.close()
            if raw_client is not None:
                raw_client.sock.close(0)


class DashboardHandler(BaseHTTPRequestHandler):
    controller: DemoController
    state: DemoState
    index_path: Path

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length < 0 or length > 4096:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    @staticmethod
    def _require_task_id(payload: dict[str, Any]) -> int:
        if "task_id" not in payload:
            raise ValueError("request body must include task_id")
        return validate_libero_object_task_id(payload["task_id"])

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            body = self.index_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            self._send_json(self.state.snapshot())
            return
        if path == "/api/frame.jpg":
            body = self.state.frame()
            if not body:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/start":
            try:
                payload = self._read_json()
                task_id = self._require_task_id(payload)
                started = self.controller.start(task_id)
            except ValueError as error:
                self._send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            status = HTTPStatus.OK if started else HTTPStatus.CONFLICT
            response: dict[str, Any] = {"ok": started, "state": self.state.snapshot()}
            if not started:
                response["error"] = "a demo rollout is already running"
            self._send_json(response, status)
            return
        if path == "/api/stop":
            stopping = self.controller.stop()
            self._send_json({"ok": stopping, "state": self.state.snapshot()})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omniinfer-url", default="http://127.0.0.1:9000")
    parser.add_argument("--backend", default="vla.cpp-linux-cuda")
    parser.add_argument("--model", help="VLA checkpoint path; omit to use the loaded runtime")
    parser.add_argument("--mmproj")
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="vla-server launch arg; repeat and use --server-arg=--flag for leading dashes",
    )
    parser.add_argument(
        "--admin-api-key-env",
        default="OMNIINFER_ADMIN_API_KEY",
        help="Environment variable containing an optional OmniInfer admin API key",
    )
    parser.add_argument(
        "--protoc",
        help="Path to protoc when protobuf-compiler is not available on PATH",
    )
    parser.add_argument("--arch", choices=SUPPORTED_DEMO_ARCHES, default="smolvla")
    parser.add_argument("--tokenizer")
    parser.add_argument("--stats-json")
    parser.add_argument("--unnorm-key")
    parser.add_argument("--task", choices=["libero_object"], default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/vla-libero")
    parser.add_argument("--view-mode", choices=["single-view", "multi-view"], default="multi-view")
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--recv-timeout-ms", type=int, default=120_000)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=7860)
    parser.add_argument(
        "--auto-start", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    try:
        validate_libero_object_task_id(args.task_id)
    except ValueError as error:
        parser.error(str(error))
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.n_action_steps < 1:
        parser.error("--n-action-steps must be >= 1")
    if not (1 <= args.listen_port <= 65535):
        parser.error("--listen-port must be between 1 and 65535")
    if args.listen_host not in LOOPBACK_HOSTS:
        parser.error("--listen-host must be a loopback address; use SSH port forwarding for remote access")
    return args


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    config = DemoConfig(
        omniinfer_url=args.omniinfer_url,
        backend=args.backend,
        model=args.model,
        mmproj=args.mmproj,
        launch_args=tuple(args.server_arg),
        admin_api_key=os.environ.get(args.admin_api_key_env),
        protoc=args.protoc,
        arch=args.arch,
        tokenizer=args.tokenizer,
        stats_json=args.stats_json,
        unnorm_key=args.unnorm_key,
        task=args.task,
        task_id=args.task_id,
        episodes=args.episodes,
        seed=args.seed,
        fps=args.fps,
        output_dir=args.output_dir,
        view_mode=args.view_mode,
        n_action_steps=args.n_action_steps,
        recv_timeout_ms=args.recv_timeout_ms,
    )
    state = DemoState(config)
    controller = DemoController(config, state, repository_root)
    DashboardHandler.controller = controller
    DashboardHandler.state = state
    DashboardHandler.index_path = Path(__file__).with_name("index.html")

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), DashboardHandler)
    print(f"VLA LIBERO demo: http://{args.listen_host}:{args.listen_port}", flush=True)
    if args.listen_host in LOOPBACK_HOSTS:
        print(
            f"Remote host: ssh -L {args.listen_port}:127.0.0.1:{args.listen_port} <host>",
            flush=True,
        )
    if args.auto_start:
        controller.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
