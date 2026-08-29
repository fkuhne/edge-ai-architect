"""Tests for ground rule 2: nothing platform-specific escapes its module.

These are the tests worth having. A benchmark harness that quietly stops being
portable is worse than one that was never portable, because the results keep
looking valid.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import runners
from edgebench import device, power

REPO = Path(__file__).resolve().parent.parent

#: Modules permitted to touch platform-specific APIs. Everything else must go
#: through `device.resolve()`, `power.get_sampler()`, or `runners`.
PLATFORM_ALLOWLIST = {
    "edgebench/power/powermetrics.py",
    "edgebench/power/nvml.py",
    "edgebench/power/rapl.py",
    "runners/coreml_runner.py",
    "runners/mlx_runner.py",
}

FORBIDDEN_IMPORTS = {"coremltools", "mlx", "pynvml", "tensorflow", "keras"}


def _python_files() -> list[Path]:
    return [
        p
        for d in ("edgebench", "runners", "projects")
        for p in (REPO / d).rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_no_platform_imports_outside_allowlist():
    """Apple- and NVIDIA-specific packages stay behind their plug-ins."""
    violations = []
    for path in _python_files():
        rel = path.relative_to(REPO).as_posix()
        if rel in PLATFORM_ALLOWLIST:
            continue
        leaked = _imported_names(path) & FORBIDDEN_IMPORTS
        if leaked:
            violations.append(f"{rel}: {sorted(leaked)}")
    assert not violations, "platform-specific imports escaped their module:\n" + "\n".join(violations)


def test_no_tensorflow_anywhere():
    """Ground rule 1. TensorFlow is out of scope, allowlist included."""
    for path in _python_files():
        leaked = _imported_names(path) & {"tensorflow", "keras", "tflite_runtime"}
        assert not leaked, f"{path.relative_to(REPO)} imports {sorted(leaked)}"


def test_cpu_is_always_available():
    assert "cpu" in device.available()
    assert device.resolve("cpu") == "cpu"


def test_unavailable_device_raises_rather_than_downgrading():
    """A silent downgrade would make every affected benchmark a lie."""
    with pytest.raises(RuntimeError, match="not available"):
        device.resolve("not-a-real-device")


def test_device_info_populated_for_every_available_device():
    for kind in device.available():
        info = device.info(kind)
        assert info.kind == kind
        assert info.name


def test_power_always_resolves_to_something():
    """Never raises on an unsupported platform; NullSampler is the floor."""
    sampler = power.get_sampler()
    assert sampler.name in power.available()
    sampler.start()
    reading = sampler.stop()
    assert reading.duration_s >= 0


def test_power_unknown_sampler_falls_back_quietly():
    assert power.get_sampler("no-such-sampler").name == "null"


def test_null_sampler_reports_unavailable_rather_than_zero():
    """`None` means "not measured". Zero watts would be a false measurement."""
    sampler = power.get_sampler("null")
    sampler.start()
    reading = sampler.stop()
    assert reading.avg_power_w is None
    assert not reading.available


@pytest.mark.parametrize("name", sorted(runners.registered()))
def test_backend_is_available_never_raises(name):
    """Discovery runs on every platform; an unusable backend is False, not a crash."""
    assert isinstance(runners.get(name).is_available(), bool)


def test_platform_specific_backends_absent_off_darwin():
    for name in ("coreml", "mlx"):
        if sys.platform != "darwin":
            assert not runners.get(name).is_available()


def test_registry_reports_all_four_backends():
    assert set(runners.registered()) == {"torch", "onnx", "coreml", "mlx"}
