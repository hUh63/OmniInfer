import importlib.util
import http.client
import json
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = REPOSITORY_ROOT / "examples" / "vla-libero" / "demo.py"
if not DEMO_PATH.is_file():
    DEMO_PATH = Path(__file__).with_name("demo.py")
SPEC = importlib.util.spec_from_file_location("vla_libero_demo", DEMO_PATH)
DEMO = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = DEMO
SPEC.loader.exec_module(DEMO)


class RuntimeContractTests(unittest.TestCase):
    def test_demo_limits_architectures_to_validated_request_formats(self):
        self.assertEqual(DEMO.SUPPORTED_DEMO_ARCHES, ("smolvla", "pi05"))

    def test_pi05_accepts_official_defaults_or_explicit_local_stats(self):
        self.assertIsNone(DEMO.validate_arch_options("pi05", None, None))
        with tempfile.NamedTemporaryFile() as stats:
            self.assertEqual(
                DEMO.validate_arch_options("pi05", "local-tokenizer", stats.name),
                stats.name,
            )

    def test_pi05_rejects_a_missing_explicit_stats_file(self):
        with self.assertRaisesRegex(ValueError, "--stats-json must be an existing file"):
            DEMO.validate_arch_options("pi05", None, "/missing/meta/stats.json")

    def test_pi05_explicit_stats_path_expands_the_user_directory(self):
        with tempfile.TemporaryDirectory() as home:
            stats = Path(home) / "stats.json"
            stats.write_text("{}")
            with mock.patch.dict(DEMO.os.environ, {"HOME": home}):
                self.assertEqual(
                    DEMO.validate_arch_options("pi05", None, "~/stats.json"),
                    str(stats),
                )

    def test_smolvla_accepts_tokenizer_override_but_rejects_pi05_stats(self):
        DEMO.validate_arch_options("smolvla", "local-tokenizer", None)
        with self.assertRaisesRegex(ValueError, "--stats-json is only supported"):
            DEMO.validate_arch_options("smolvla", None, "stats.json")

    def test_create_policy_forwards_pi05_client_options(self):
        client_package = types.ModuleType("client")
        client_module = types.ModuleType("client.vla_cpp_client")
        raw_client = mock.Mock()
        client_module.VlaCppClient = mock.Mock(return_value=raw_client)
        client_package.vla_cpp_client = client_module
        config = DEMO.DemoConfig(
            arch="pi05",
            tokenizer="local-tokenizer",
            stats_json="stats.json",
            n_action_steps=10,
        )
        with mock.patch.dict(
            sys.modules,
            {"client": client_package, "client.vla_cpp_client": client_module},
        ):
            created, adapter = DEMO.create_policy(config, "tcp://127.0.0.1:5555")
        self.assertIs(created, raw_client)
        self.assertIs(adapter._client, raw_client)
        client_module.VlaCppClient.assert_called_once_with(
            vla_addr="tcp://127.0.0.1:5555",
            arch="pi05",
            tokenizer_name="local-tokenizer",
            recv_timeout_ms=120_000,
            n_action_steps=10,
            stats_json="stats.json",
        )

    def test_model_profiles_map_server_paths_without_exposing_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smolvla = root / "smolvla.gguf"
            pi05 = root / "pi05.gguf"
            stats = root / "stats.json"
            tokenizer = root / "tokenizer"
            for path in (smolvla, pi05, stats):
                path.write_text("test")
            tokenizer.mkdir()
            manifest = root / "models.json"
            manifest.write_text(
                json.dumps(
                    {
                        "models": {
                            "smol": {
                                "label": "SmolVLA demo",
                                "arch": "smolvla",
                                "model": smolvla.name,
                            },
                            "pi": {
                                "label": "PI0.5 demo",
                                "arch": "pi05",
                                "model": pi05.name,
                                "tokenizer": str(tokenizer),
                                "stats_json": stats.name,
                            },
                        }
                    }
                )
            )
            profiles = DEMO.load_model_profiles(str(manifest), DEMO.DemoConfig())
            self.assertEqual(list(profiles), ["smol", "pi"])
            self.assertEqual(profiles["smol"].config.model, str(smolvla.resolve()))
            self.assertEqual(profiles["pi"].config.stats_json, str(stats.resolve()))
            self.assertEqual(profiles["pi"].config.tokenizer, str(tokenizer.resolve()))
            self.assertEqual(profiles["pi"].config.n_action_steps, 10)
            public = [profile.public() for profile in profiles.values()]
            self.assertEqual(
                public,
                [
                    {"id": "smol", "label": "SmolVLA demo", "arch": "smolvla"},
                    {"id": "pi", "label": "PI0.5 demo", "arch": "pi05"},
                ],
            )
            self.assertNotIn(str(root), json.dumps(public))

    def test_model_profiles_preserve_hugging_face_tokenizer_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "pi05.gguf"
            model.write_text("test")
            manifest = root / "models.json"
            manifest.write_text(
                json.dumps(
                    {
                        "models": {
                            "pi": {
                                "label": "PI0.5",
                                "arch": "pi05",
                                "model": model.name,
                                "tokenizer": "google/paligemma-3b-pt-224",
                            }
                        }
                    }
                )
            )
            profiles = DEMO.load_model_profiles(str(manifest), DEMO.DemoConfig())
            self.assertEqual(
                profiles["pi"].config.tokenizer,
                "google/paligemma-3b-pt-224",
            )

    def test_model_profiles_reject_unknown_fields_and_missing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for entry, message in (
                (
                    {"label": "bad", "arch": "smolvla", "model": "missing.gguf"},
                    "model does not exist",
                ),
                (
                    {
                        "label": "bad",
                        "arch": "smolvla",
                        "model": "missing.gguf",
                        "arbitrary_path": "/secret",
                    },
                    "unknown field",
                ),
            ):
                with self.subTest(message=message):
                    manifest = root / "models.json"
                    manifest.write_text(json.dumps({"models": {"bad": entry}}))
                    with self.assertRaisesRegex(ValueError, message):
                        DEMO.load_model_profiles(str(manifest), DEMO.DemoConfig())

    def test_dashboard_state_exposes_profile_label_not_model_path(self):
        config = DEMO.DemoConfig(model="/private/models/smolvla.gguf")
        profiles = {
            "smol": DEMO.ModelProfile("smol", "SmolVLA demo", config)
        }
        snapshot = DEMO.DemoState(config, profiles, "smol").snapshot()
        serialized = json.dumps(snapshot)
        self.assertEqual(snapshot["model"], "SmolVLA demo")
        self.assertNotIn("/private/models", serialized)

    def test_profile_errors_redact_server_paths(self):
        config = DEMO.DemoConfig(
            model="/private/models/pi05.gguf",
            tokenizer="/private/tokenizer",
            stats_json="/private/stats.json",
        )
        error = RuntimeError(
            "failed /private/models/pi05.gguf using /private/stats.json"
        )
        public = DEMO.public_profile_error(error, config)
        self.assertNotIn("/private/", public)
        self.assertIn("<model-profile-value>", public)

    def test_dashboard_is_loopback_only_and_idle_by_default(self):
        source = DEMO_PATH.read_text()
        self.assertIn('LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}', source)
        self.assertIn('default=False', source)
        self.assertIn('must be a loopback address', source)

    def test_readme_has_copyable_linux_quick_start(self):
        readme = (REPOSITORY_ROOT / "examples" / "vla-libero" / "README.md").read_text()
        self.assertIn("git submodule update --init framework/vla.cpp", readme)
        self.assertIn("apt-get install -y git protobuf-compiler", readme)
        self.assertIn("astral.sh/uv/install.sh", readme)
        self.assertIn("examples/vla-libero/setup.sh", readme)
        self.assertIn("examples/vla-libero/run.sh", readme)

    def test_readme_describes_the_end_to_end_simulation_flow(self):
        readme = (REPOSITORY_ROOT / "examples" / "vla-libero" / "README.md").read_text()
        self.assertIn("## Simulation flow", readme)
        self.assertIn("LIBERO/MuJoCo resets", readme)
        self.assertIn("ZeroMQ/protobuf endpoint", readme)
        self.assertIn("success`, `failed`, `partial`, `stopped`, or `error`", readme)

    def test_pi05_tokenizer_runtime_dependency_is_pinned(self):
        requirements = (
            REPOSITORY_ROOT / "examples" / "vla-libero" / "requirements.txt"
        ).read_text()
        self.assertIn("sentencepiece==0.2.0", requirements.splitlines())

    def test_readme_marks_pi05_experimental_and_explains_runtime_cleanup(self):
        readme = (REPOSITORY_ROOT / "examples" / "vla-libero" / "README.md").read_text()
        self.assertIn("is an experimental request path", readme)
        self.assertIn("PI0.5 checkpoint rollout result", readme)
        self.assertIn("does\nnot unload the previous runtime", readme)
        self.assertIn("POST /omni/model/unload", readme)
        self.assertIn("409 model_reload_required", readme)

    def test_rollout_video_directory_uses_a_unique_run_identity(self):
        self.assertIn('time.time_ns()', DEMO_PATH.read_text())
    def test_setup_defaults_to_cpu_torch_and_exposes_cuda_override(self):
        setup = (REPOSITORY_ROOT / "examples" / "vla-libero" / "setup.sh").read_text()
        self.assertIn('TORCH_BACKEND="cpu"', setup)
        self.assertIn('--torch-backend "$TORCH_BACKEND"', setup)
        self.assertIn('omniinfer_libero_source.pth', setup)
        self.assertIn('"$HOME/.local/bin/uv"', setup)
        self.assertIn('LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"', setup)
        self.assertIn('LIBERO_DIR="$DEMO_CACHE_DIR/LIBERO"', setup)

    def test_readme_requires_isolated_gateway_roots_on_shared_hosts(self):
        readme = (REPOSITORY_ROOT / "examples" / "vla-libero" / "README.md").read_text()
        self.assertIn('--state-root "$DEMO_ROOT/state"', readme)
        self.assertIn('--runtime-root "$DEMO_ROOT/runtimes"', readme)
        self.assertIn("changing only", readme)
        self.assertIn("the port still shares OmniInfer state", readme)
        self.assertIn(
            "scripts/platforms/linux/vla.cpp-linux-cuda/build.sh --from-source",
            readme,
        )
        self.assertIn(
            'cp -a .local/runtime/linux/vla.cpp-linux-cuda "$DEMO_ROOT/runtimes/"',
            readme,
        )

    def test_readme_explains_torch_backend_changes_need_a_fresh_venv(self):
        readme = (REPOSITORY_ROOT / "examples" / "vla-libero" / "README.md").read_text()
        self.assertIn("does not convert an", readme)
        self.assertIn("existing venv", readme)
        self.assertIn("vla-libero-demo/venv-cu124", readme)

    def test_setup_disables_shared_robosuite_file_logging_and_smokes_import(self):
        setup = (
            REPOSITORY_ROOT / "examples" / "vla-libero" / "setup.sh"
        ).read_text()
        self.assertIn('FILE_LOGGING_LEVEL = None', setup)
        self.assertIn("PYTHONPATH=\"$VLA_CPP_ROOT/eval\"", setup)
        self.assertIn("import gymnasium; import sim.libero", setup)

    def test_model_profile_example_exists_and_contains_both_architectures(self):
        example = (
            REPOSITORY_ROOT
            / "examples"
            / "vla-libero"
            / "model-profiles.example.json"
        )
        payload = json.loads(example.read_text())
        self.assertEqual(
            {entry["arch"] for entry in payload["models"].values()},
            {"smolvla", "pi05"},
        )

    def test_generated_output_defaults_to_per_user_state_directory(self):
        with mock.patch.dict(DEMO.os.environ, {"XDG_STATE_HOME": "/tmp/state"}):
            self.assertEqual(
                DEMO.default_output_dir(),
                "/tmp/state/omniinfer/vla-libero-demo/outputs",
            )

    def test_csrf_token_is_required_for_mutations(self):
        DEMO.validate_csrf_token("expected", "expected")
        for received in (None, "", "wrong"):
            with self.subTest(received=received):
                with self.assertRaises(PermissionError):
                    DEMO.validate_csrf_token("expected", received)

    def test_dashboard_injects_and_returns_csrf_token(self):
        source = DEMO_PATH.read_text()
        page = (REPOSITORY_ROOT / "examples" / "vla-libero" / "index.html").read_text()
        self.assertIn('secrets.token_urlsafe(32)', source)
        self.assertIn('X-OmniInfer-CSRF-Token', source)
        self.assertIn('{{CSRF_TOKEN}}', page)
        self.assertIn('X-OmniInfer-CSRF-Token', page)

    def test_dashboard_rejects_post_without_csrf_token(self):
        class Controller:
            def __init__(self):
                self.started = False

            def start(self, task_id, model_profile):
                self.started = True
                self.model_profile = model_profile
                return True

            def stop(self):
                return True

        controller = Controller()
        state = DEMO.DemoState(DEMO.DemoConfig())
        DEMO.DashboardHandler.controller = controller
        DEMO.DashboardHandler.state = state
        DEMO.DashboardHandler.index_path = (
            REPOSITORY_ROOT / "examples" / "vla-libero" / "index.html"
        )
        DEMO.DashboardHandler.csrf_token = "test-token"
        server = ThreadingHTTPServer(("127.0.0.1", 0), DEMO.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            origin = f"http://127.0.0.1:{server.server_address[1]}"
            connection.request("GET", "/")
            page_response = connection.getresponse()
            page = page_response.read().decode("utf-8")
            self.assertEqual(page_response.status, 200)
            self.assertIn('content="test-token"', page)
            self.assertNotIn("{{CSRF_TOKEN}}", page)
            self.assertEqual(page_response.getheader("X-Frame-Options"), "DENY")
            connection.close()

            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            connection.request(
                "POST",
                "/api/start",
                body='{"task_id":0,"model_profile":"command-line"}',
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            self.assertIn("CSRF", json.loads(response.read())["error"])
            self.assertFalse(controller.started)
            connection.close()

            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            connection.request(
                "POST",
                "/api/start",
                body='{"task_id":0,"model_profile":"command-line"}',
                headers={
                    "Content-Type": "application/json",
                    "Origin": origin,
                    "X-OmniInfer-CSRF-Token": "test-token",
                },
            )
            self.assertEqual(connection.getresponse().status, 200)
            self.assertTrue(controller.started)
            self.assertEqual(controller.model_profile, "command-line")
            connection.close()

            controller.started = False
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            connection.request(
                "POST",
                "/api/start",
                body=(
                    '{"task_id":0,"model_profile":"command-line",'
                    '"model":"/private/injected.gguf"}'
                ),
                headers={
                    "Content-Type": "application/json",
                    "Origin": origin,
                    "X-OmniInfer-CSRF-Token": "test-token",
                },
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn("unexpected field", json.loads(response.read())["error"])
            self.assertFalse(controller.started)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_dashboard_rejects_dns_rebinding_and_foreign_origins(self):
        class Controller:
            started = False

            def start(self, task_id, model_profile):
                self.started = True
                return True

            def stop(self):
                return True

        controller = Controller()
        DEMO.DashboardHandler.controller = controller
        DEMO.DashboardHandler.state = DEMO.DemoState(DEMO.DemoConfig())
        DEMO.DashboardHandler.index_path = (
            REPOSITORY_ROOT / "examples" / "vla-libero" / "index.html"
        )
        DEMO.DashboardHandler.csrf_token = "test-token"
        server = ThreadingHTTPServer(("127.0.0.1", 0), DEMO.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/", headers={"Host": "attacker.example"})
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            self.assertNotIn("test-token", response.read().decode("utf-8"))
            connection.close()

            for host, origin in (
                ("attacker.example", "http://attacker.example"),
                (f"127.0.0.1:{port}", "https://evil.example"),
                (f"127.0.0.1:{port}", None),
            ):
                with self.subTest(host=host, origin=origin):
                    headers = {
                        "Host": host,
                        "Content-Type": "application/json",
                        "X-OmniInfer-CSRF-Token": "test-token",
                    }
                    if origin is not None:
                        headers["Origin"] = origin
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", port, timeout=2
                    )
                    connection.request(
                        "POST",
                        "/api/start",
                        body='{"task_id":0,"model_profile":"command-line"}',
                        headers=headers,
                    )
                    self.assertEqual(connection.getresponse().status, 403)
                    connection.close()
            self.assertFalse(controller.started)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_dashboard_requires_json_content_type(self):
        handler = object.__new__(DEMO.DashboardHandler)
        handler.headers = {"Content-Length": "2", "Content-Type": "text/plain"}
        with self.assertRaisesRegex(ValueError, "Content-Type"):
            handler._read_json()

    def test_run_uses_the_isolated_demo_environment(self):
        runner = (REPOSITORY_ROOT / "examples" / "vla-libero" / "run.sh").read_text()
        self.assertIn('LIBERO_CONFIG_PATH', runner)
        self.assertIn('"$VENV_DIR/bin/python" "$SCRIPT_DIR/demo.py"', runner)
        self.assertIn('currently supports Linux only', runner)

    def test_shell_wrappers_report_missing_option_values(self):
        for script, option in (("setup.sh", "--venv"), ("run.sh", "--libero-config")):
            with self.subTest(script=script, option=option):
                result = subprocess.run(
                    ["bash", str(REPOSITORY_ROOT / "examples" / "vla-libero" / script), option],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"{option} requires a value", result.stderr)
                self.assertNotIn("unbound variable", result.stderr)

    def test_accepts_all_libero_object_task_ids(self):
        for task_id in range(10):
            self.assertEqual(DEMO.validate_libero_object_task_id(task_id), task_id)

    def test_rejects_invalid_libero_object_task_ids(self):
        for task_id in (-1, 10, True, "1", 1.0):
            with self.subTest(task_id=task_id):
                with self.assertRaises(ValueError):
                    DEMO.validate_libero_object_task_id(task_id)

    def test_completed_result_preserves_mixed_outcomes(self):
        self.assertEqual(DEMO.aggregate_result(3, 0), "success")
        self.assertEqual(DEMO.aggregate_result(0, 3), "failed")
        self.assertEqual(DEMO.aggregate_result(2, 1), "partial")

    def test_accepts_managed_loopback_vla_runtime(self):
        endpoint, backend, model = DEMO.validate_vla_runtime(
            {
                "external_server_protocol": "vla.cpp-zmq-server",
                "client_endpoint": "tcp://127.0.0.1:15555",
                "selected_backend": "vla.cpp-linux-cuda",
                "selected_model": "/models/smolvla.gguf",
            }
        )
        self.assertEqual(endpoint, "tcp://127.0.0.1:15555")
        self.assertEqual(backend, "vla.cpp-linux-cuda")
        self.assertEqual(model, "/models/smolvla.gguf")

    def test_rejects_openai_runtime_protocol(self):
        with self.assertRaisesRegex(ValueError, "expected 'vla.cpp-zmq-server'"):
            DEMO.validate_vla_runtime(
                {
                    "external_server_protocol": "llama.cpp-server",
                    "client_endpoint": "http://127.0.0.1:8080",
                    "backend": "llama.cpp-linux-cuda",
                }
            )

    def test_rejects_non_loopback_vla_endpoint(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            DEMO.validate_vla_runtime(
                {
                    "external_server_protocol": "vla.cpp-zmq-server",
                    "client_endpoint": "tcp://192.0.2.10:5555",
                    "backend": "vla.cpp-linux-cuda",
                }
            )

    def test_model_load_payload_preserves_vla_contract(self):
        config = DEMO.DemoConfig(
            model="/models/smolvla.gguf",
            mmproj="/models/mmproj.gguf",
            launch_args=("--timing-detail", "phase"),
        )
        self.assertEqual(
            config.model_load_payload(),
            {
                "model": "/models/smolvla.gguf",
                "backend": "vla.cpp-linux-cuda",
                "strict_capabilities": True,
                "mmproj": "/models/mmproj.gguf",
                "launch_args": ["--timing-detail", "phase"],
            },
        )

    def test_no_model_uses_existing_managed_runtime(self):
        self.assertIsNone(DEMO.DemoConfig(model=None).model_load_payload())

    def test_configure_protoc_rejects_non_executable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "protoc"
            candidate.write_text("not executable")
            with self.assertRaisesRegex(ValueError, "not an executable"):
                DEMO.configure_protoc(str(candidate))


class MetricTests(unittest.TestCase):
    def test_start_requires_an_explicit_task_id(self):
        with self.assertRaisesRegex(ValueError, "include task_id"):
            DEMO.DashboardHandler._require_task_id({})
        self.assertEqual(DEMO.DashboardHandler._require_task_id({"task_id": 4}), 4)

    def test_start_requires_a_model_profile_id(self):
        with self.assertRaisesRegex(ValueError, "include model_profile"):
            DEMO.DashboardHandler._require_model_profile({})
        self.assertEqual(
            DEMO.DashboardHandler._require_model_profile({"model_profile": "pi05"}),
            "pi05",
        )

    def test_controller_freezes_selected_task_while_running(self):
        entered = threading.Event()

        class RecordingController(DEMO.DemoController):
            def _run(self, task_id, profile):
                self.recorded_task_id = task_id
                entered.set()
                self._stop.wait(2)

        state = DEMO.DemoState(DEMO.DemoConfig())
        controller = RecordingController(DEMO.DemoConfig(), state, REPOSITORY_ROOT)
        self.assertTrue(controller.start(7))
        self.assertTrue(entered.wait(1))
        self.assertEqual(controller.recorded_task_id, 7)
        self.assertEqual(state.snapshot()["task_id"], 7)
        self.assertFalse(controller.start(2))
        self.assertTrue(controller.stop())

    def test_controller_rejects_unknown_profile_and_freezes_selected_profile(self):
        entered = threading.Event()

        class RecordingController(DEMO.DemoController):
            def _run(self, task_id, profile):
                self.recorded_profile = profile.identifier
                entered.set()
                self._stop.wait(2)

        config = DEMO.DemoConfig(model="/models/smol.gguf")
        profiles = {
            "smol": DEMO.ModelProfile("smol", "SmolVLA", config),
            "pi": DEMO.ModelProfile(
                "pi", "PI0.5", DEMO.replace(config, arch="pi05", n_action_steps=10)
            ),
        }
        state = DEMO.DemoState(config, profiles, "smol")
        controller = RecordingController(
            config, state, REPOSITORY_ROOT, profiles, "smol"
        )
        with self.assertRaisesRegex(ValueError, "unknown model_profile"):
            controller.start(0, "missing")
        self.assertTrue(controller.start(0, "pi"))
        self.assertTrue(entered.wait(1))
        self.assertEqual(controller.recorded_profile, "pi")
        self.assertEqual(state.snapshot()["model_profile"], "pi")
        self.assertFalse(controller.start(0, "smol"))
        self.assertTrue(controller.stop())

    def test_begin_clears_previous_rollout_identity(self):
        state = DEMO.DemoState(DEMO.DemoConfig())
        state.update(
            task_description="previous task",
            client_endpoint="tcp://127.0.0.1:15555",
        )
        state.begin(4)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["task_id"], 4)
        self.assertEqual(
            snapshot["task_description"],
            "pick up the ketchup and place it in the basket",
        )
        self.assertIsNone(snapshot["client_endpoint"])

    def test_state_exposes_only_the_ten_predefined_object_tasks(self):
        state = DEMO.DemoState(DEMO.DemoConfig())
        options = state.snapshot()["task_options"]
        self.assertEqual(len(options), 10)
        self.assertEqual([option["task_id"] for option in options], list(range(10)))

    def test_empty_metric_summary(self):
        self.assertEqual(
            DEMO.metric_summary([]),
            {"samples": 0, "last_ms": None, "mean_ms": None, "p50_ms": None, "p95_ms": None},
        )

    def test_metric_summary_uses_interpolated_percentiles(self):
        summary = DEMO.metric_summary([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["last_ms"], 40.0)
        self.assertEqual(summary["mean_ms"], 25.0)
        self.assertEqual(summary["p50_ms"], 25.0)
        self.assertEqual(summary["p95_ms"], 38.5)

    def test_dashboard_separates_prediction_from_queue_replay(self):
        state = DEMO.DemoState(DEMO.DemoConfig())
        state.begin(0)
        state.publish_step(
            action=[0.1] * 7,
            policy_ms=50.0,
            prediction_sent=True,
            env_ms=3.0,
            loop_ms=53.0,
            reward=0.0,
            step=1,
        )
        state.publish_step(
            action=[0.2] * 7,
            policy_ms=0.2,
            prediction_sent=False,
            env_ms=3.1,
            loop_ms=3.3,
            reward=0.1,
            step=2,
        )
        snapshot = state.snapshot()
        self.assertEqual(snapshot["latency"]["policy"]["samples"], 2)
        self.assertEqual(snapshot["latency"]["prediction"]["samples"], 1)
        self.assertEqual(snapshot["latency"]["prediction"]["last_ms"], 50.0)
        self.assertEqual(snapshot["call_kind"], "action_queue_replay")
        self.assertEqual(snapshot["action"], [0.2] * 7)


if __name__ == "__main__":
    unittest.main()
