pub(super) const ROCM_PLATFORM_PLUGIN: &str = r#"import os

_shim_active = False
_uva_fallback_active = False


def _is_wsl():
    try:
        with open("/proc/sys/kernel/osrelease", encoding="utf-8") as release:
            return "microsoft" in release.read().lower()
    except OSError:
        return False


def _amdsmi_has_gpu():
    initialized = False
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        initialized = True
        return bool(amdsmi.amdsmi_get_processor_handles())
    except Exception:
        return False
    finally:
        if initialized:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass


def _install_amdsmi_shim(devices):
    import amdsmi

    def handles():
        return list(range(len(devices)))

    def properties(handle):
        return devices[int(handle)]

    def asic_info(handle):
        device = properties(handle)
        return {
            "asic_serial": device["asic_serial"],
            "device_id": "",
            "market_name": device["name"],
            "target_graphics_version": device["gcn_arch"],
        }

    amdsmi.amdsmi_init = lambda *args, **kwargs: None
    amdsmi.amdsmi_shut_down = lambda: None
    amdsmi.amdsmi_get_processor_handles = handles
    amdsmi.amdsmi_get_gpu_asic_info = asic_info
    amdsmi.amdsmi_get_gpu_memory_total = (
        lambda handle, memory_type: properties(handle)["total_memory"]
    )
    amdsmi.amdsmi_get_gpu_device_uuid = (
        lambda handle: properties(handle)["uuid"]
    )
    amdsmi.amdsmi_topo_get_link_type = (
        lambda handle, peer_handle: {"hops": 1, "type": 2}
    )
    amdsmi.amdsmi_topo_get_numa_node_number = lambda handle: 0


def platform_plugin():
    global _shim_active
    if _shim_active:
        return "vllm.platforms.rocm.RocmPlatform"
    if os.environ.get("HSA_ENABLE_DXG_DETECTION") != "1" or not _is_wsl():
        return None
    if _amdsmi_has_gpu():
        return None
    try:
        import torch

        if (
            torch.version.hip
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 0
        ):
            devices = []
            for index in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(index)
                uuid = str(getattr(device, "uuid", f"wsl2-rocm-{index}"))
                devices.append(
                    {
                        "asic_serial": f"0x{uuid.replace('-', '')}",
                        "gcn_arch": device.gcnArchName,
                        "name": device.name,
                        "total_memory": device.total_memory,
                        "uuid": uuid,
                    }
                )
            _install_amdsmi_shim(devices)
            _shim_active = True
            return "vllm.platforms.rocm.RocmPlatform"
    except Exception:
        pass
    return None


def general_plugin():
    global _uva_fallback_active
    if _uva_fallback_active:
        return
    if not _shim_active:
        platform_plugin()
    if not _shim_active:
        return

    from vllm.utils.platform_utils import is_uva_available

    if is_uva_available():
        return

    import torch
    from vllm.v1.worker.gpu import buffer_utils

    class WslRocmBuffer:
        def __init__(self, size, dtype):
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
            self.np = self.cpu.numpy()
            self.uva = torch.zeros(size, dtype=dtype, device="cuda")

    def copy_to_accelerator(self, value):
        self._curr = (self._curr + 1) % self.max_concurrency
        buffer = self._uva_bufs[self._curr]
        destination = buffer.cpu if isinstance(value, torch.Tensor) else buffer.np
        count = len(value)
        destination[:count] = value
        return buffer.uva[:count].copy_(buffer.cpu[:count])

    buffer_utils.UvaBuffer = WslRocmBuffer
    buffer_utils.UvaBufferPool.copy_to_uva = copy_to_accelerator
    _uva_fallback_active = True
"#;

pub(super) const ROCM_PLATFORM_PLUGIN_ENTRY_POINTS: &str = r#"[vllm.platform_plugins]
omniinfer_wsl2_rocm = omniinfer_vllm_wsl2_rocm:platform_plugin

[vllm.general_plugins]
omniinfer_wsl2_rocm = omniinfer_vllm_wsl2_rocm:general_plugin
"#;

pub(super) const RUNNER_SCRIPT: &str = r#"#!/bin/sh
set -eu
pid_file=$1
shift
runtime_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -f "$runtime_dir/runtime.env" ]; then
    set -a
    . "$runtime_dir/runtime.env"
    set +a
fi
mkdir -p "$(dirname "$pid_file")"
if [ -s "$pid_file" ]; then
    old_pid=$(cat "$pid_file")
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "vLLM runtime is already active with pid $old_pid" >&2
        exit 73
    fi
    rm -f "$pid_file"
fi
managed_memory_policy=1
managed_eager_policy=1
managed_chunked_prefill_policy=1
for argument in "$@"; do
    case "$argument" in
        --kv-cache-memory-bytes|--kv-cache-memory-bytes=*|--gpu-memory-utilization|--gpu-memory-utilization=*)
            managed_memory_policy=0
            ;;
        --enforce-eager|--no-enforce-eager)
            managed_eager_policy=0
            ;;
        --enable-chunked-prefill|--enable-chunked-prefill=*|--no-enable-chunked-prefill)
            managed_chunked_prefill_policy=0
            ;;
    esac
done
if [ "${HSA_ENABLE_DXG_DETECTION:-}" = "1" ]; then
    if [ "$managed_memory_policy" -eq 1 ]; then
        memory_kib=$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
        case "$memory_kib" in
            ''|*[!0-9]*) memory_kib=0 ;;
        esac
        kv_cache_bytes=$((memory_kib * 1024 / 5))
        if [ "$kv_cache_bytes" -gt 4294967296 ]; then
            kv_cache_bytes=4294967296
        fi
        if [ "$kv_cache_bytes" -ge 268435456 ]; then
            set -- "$@" --kv-cache-memory-bytes "$kv_cache_bytes"
            echo "OmniInfer: limiting WSL2 ROCm KV cache to $kv_cache_bytes bytes based on Linux memory; override with --kv-cache-memory-bytes or --gpu-memory-utilization" >&2
        fi
    fi
    if [ "$managed_eager_policy" -eq 1 ]; then
        set -- "$@" --enforce-eager
    fi
    if [ "$managed_chunked_prefill_policy" -eq 1 ]; then
        set -- "$@" --no-enable-chunked-prefill
    fi
    echo "OmniInfer: applying WSL2 ROCm compatibility defaults for eager execution and non-chunked prefill; explicit vLLM flags override each default" >&2
fi
unset argument managed_memory_policy managed_eager_policy managed_chunked_prefill_policy memory_kib kv_cache_bytes
setsid "$@" &
child=$!
printf '%s\n' "$child" > "$pid_file"
forward_signal() {
    kill -TERM "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
}
trap forward_signal HUP INT TERM
set +e
wait "$child"
status=$?
set -e
rm -f "$pid_file"
exit "$status"
"#;

pub(super) const STOPPER_SCRIPT: &str = r#"#!/bin/sh
set -eu
pid_file=$1
if [ ! -s "$pid_file" ]; then
    exit 0
fi
pid=$(cat "$pid_file")
case "$pid" in
    ''|*[!0-9]*)
        echo "invalid vLLM pid file: $pid_file" >&2
        exit 74
        ;;
esac
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    exit 0
fi
kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
i=0
while [ "$i" -lt 80 ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        exit 0
    fi
    i=$((i + 1))
    sleep 0.1
done
kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
rm -f "$pid_file"
"#;

pub(super) const GPU_PROBE: &str = r#"import json
import os
import torch
import vllm
from vllm.platforms import current_platform
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
expected_accelerator = os.environ["OMNIINFER_EXPECTED_ACCELERATOR"]
if expected_accelerator == "rocm" and not current_platform.is_rocm():
    raise SystemExit("vLLM did not select its ROCm platform")
if expected_accelerator == "cuda" and not current_platform.is_cuda():
    raise SystemExit("vLLM did not select its CUDA platform")
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print(json.dumps({
    "vllm_version": vllm.__version__,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "value": float(x.item()),
    "vllm_platform": type(current_platform).__module__,
}))
"#;

pub(super) const NATIVE_DEPENDENCY_PROBE: &str = r#"set -eu
runtime=$1
runtime_dir=$runtime
site_packages=
for candidate in "$runtime"/venv/lib/python*/site-packages; do
    [ -d "$candidate" ] || continue
    site_packages=$candidate
    break
done
if [ -z "$site_packages" ]; then
    echo "managed site-packages directory not found" >&2
    exit 1
fi
set -a
. "$runtime/runtime.env"
set +a
if [ -n "${CC:-}" ]; then
    [ -x "$CC" ] || {
        echo "managed C compiler is not executable: $CC" >&2
        exit 1
    }
    cc_probe="$runtime/run/.cc-probe-$$.so"
    if ! printf '%s\n' 'int omniinfer_cc_probe(void) { return 0; }' |
        "$CC" -x c -shared -fPIC -o "$cc_probe" -
    then
        echo "managed C compiler probe failed: $CC" >&2
        rm -f "$cc_probe"
        exit 1
    fi
    rm -f "$cc_probe"
fi
checked=0
missing=0
for library in \
    "$site_packages"/vllm/*.so \
    "$site_packages"/flash_attn*.so \
    "$site_packages"/aiter/jit/*.so \
    "$site_packages"/xgrammar/*.so \
    "$site_packages"/torchvision/*.so \
    "$site_packages"/torchaudio/lib/*.so
do
    [ -f "$library" ] || continue
    checked=$((checked + 1))
    unresolved=$(ldd "$library" 2>&1 | grep 'not found' || true)
    if [ -n "$unresolved" ]; then
        printf '%s\n%s\n' "$library" "$unresolved" >&2
        missing=$((missing + 1))
    fi
done
if [ "$checked" -eq 0 ]; then
    echo "no managed native extensions found" >&2
    exit 1
fi
if [ "$missing" -ne 0 ]; then
    echo "$missing of $checked managed native extensions have unresolved libraries" >&2
    exit 1
fi
printf '%s\n' "$checked"
"#;
