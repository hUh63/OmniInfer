import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path


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

    def test_rollout_video_directory_uses_a_unique_run_identity(self):
        self.assertIn('time.time_ns()', DEMO_PATH.read_text())
    def test_setup_defaults_to_cpu_torch_and_exposes_cuda_override(self):
        setup = (REPOSITORY_ROOT / "examples" / "vla-libero" / "setup.sh").read_text()
        self.assertIn('TORCH_BACKEND="cpu"', setup)
        self.assertIn('--torch-backend "$TORCH_BACKEND"', setup)
        self.assertIn('omniinfer_libero_source.pth', setup)
        self.assertIn('"$HOME/.local/bin/uv"', setup)

    def test_run_uses_the_isolated_demo_environment(self):
        runner = (REPOSITORY_ROOT / "examples" / "vla-libero" / "run.sh").read_text()
        self.assertIn('LIBERO_CONFIG_PATH', runner)
        self.assertIn('"$VENV_DIR/bin/python" "$SCRIPT_DIR/demo.py"', runner)
        self.assertIn('currently supports Linux only', runner)

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
            model="/models/pi05.gguf",
            mmproj="/models/mmproj.gguf",
            launch_args=("--timing-detail", "phase"),
        )
        self.assertEqual(
            config.model_load_payload(),
            {
                "model": "/models/pi05.gguf",
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

    def test_controller_freezes_selected_task_while_running(self):
        entered = threading.Event()

        class RecordingController(DEMO.DemoController):
            def _run(self, task_id):
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
