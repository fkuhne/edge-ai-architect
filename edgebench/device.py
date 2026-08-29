"""Runtime device resolution.

Ground rule 2 of this project: the Mac is treated as a generic Linux box. No
module outside this one may name a specific accelerator. Everything else asks
`resolve()` and gets back whatever this machine happens to have, so the same
code runs on a CUDA box without edits.

Preference order is cuda -> mps -> cpu, overridable with EDGEBENCH_DEVICE so a
run can be pinned for comparison:

    EDGEBENCH_DEVICE=cpu uv run python -m edgebench.selftest
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field

ENV_VAR = "EDGEBENCH_DEVICE"

#: Preference order. First available wins.
PREFERENCE = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class DeviceInfo:
    """What we resolved, and enough detail to make a result reproducible."""

    kind: str  # "cuda" | "mps" | "cpu"
    name: str  # human-readable chip/GPU name
    total_memory_mb: float | None = None
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        mem = f", {self.total_memory_mb:.0f} MB" if self.total_memory_mb else ""
        return f"{self.kind} ({self.name}{mem})"


def _torch():
    """Import torch lazily. edgebench core must work without it installed."""
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError:
        return None


def available() -> list[str]:
    """Device kinds this machine can actually use, in preference order."""
    found = ["cpu"]  # always true
    torch = _torch()
    if torch is not None:
        if torch.cuda.is_available():
            found.append("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            found.append("mps")
    return [d for d in PREFERENCE if d in found]


def resolve(preferred: str | None = None) -> str:
    """Return the device kind to use.

    Explicit argument wins, then EDGEBENCH_DEVICE, then preference order. An
    unavailable explicit choice is an error rather than a silent downgrade --
    a benchmark that quietly ran on the CPU is worse than one that failed.
    """
    requested = preferred or os.environ.get(ENV_VAR)
    usable = available()

    if requested:
        requested = requested.strip().lower()
        if requested not in usable:
            raise RuntimeError(
                f"device {requested!r} was requested but is not available; "
                f"this machine offers {usable}. "
                f"Unset {ENV_VAR} to auto-select."
            )
        return requested

    return usable[0]


def info(kind: str | None = None) -> DeviceInfo:
    """Describe a resolved device for the provenance record."""
    kind = kind or resolve()
    torch = _torch()

    if kind == "cuda" and torch is not None:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(
            kind="cuda",
            name=props.name,
            total_memory_mb=props.total_memory / 1024**2,
            detail={
                "capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "torch_cuda": torch.version.cuda,
            },
        )

    if kind == "mps":
        # Unified memory: the "GPU memory" is system memory, which is exactly
        # why the 8 GB ceiling on this machine bites so hard.
        return DeviceInfo(
            kind="mps",
            name=_cpu_name(),
            total_memory_mb=_system_memory_mb(),
            detail={"unified_memory": True},
        )

    return DeviceInfo(
        kind="cpu",
        name=_cpu_name(),
        total_memory_mb=_system_memory_mb(),
        detail={"cores": os.cpu_count()},
    )


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        import subprocess  # noqa: PLC0415

        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _system_memory_mb() -> float | None:
    try:
        import psutil  # noqa: PLC0415

        return psutil.virtual_memory().total / 1024**2
    except ImportError:
        return None


def torch_device(kind: str | None = None):
    """Convenience for PyTorch callers: a real `torch.device`."""
    torch = _torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed in this environment")
    return torch.device(resolve(kind))
