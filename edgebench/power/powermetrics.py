"""macOS power sampling via `powermetrics`.

Darwin-only, and an *optional plug-in* under ground rule 2 -- nothing in the
library imports this directly. It earns its place because `powermetrics` is the
only tool that reports **ANE power draw**, which is the cleanest available proof
that a model actually executed on the Neural Engine rather than silently falling
back to CPU. That check matters in Phase 2.

Requires sudo. If sudo would prompt for a password, this sampler reports itself
unavailable rather than blocking a benchmark on an interactive prompt.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile

from .base import BaseSampler, PowerReading

_SAMPLE_INTERVAL_MS = 200

# powermetrics prints e.g. "CPU Power: 1234 mW" / "ANE Power: 0 mW"
_PATTERNS = {
    "cpu": re.compile(r"^CPU Power:\s+([\d.]+)\s*mW", re.MULTILINE),
    "gpu": re.compile(r"^GPU Power:\s+([\d.]+)\s*mW", re.MULTILINE),
    "ane": re.compile(r"^ANE Power:\s+([\d.]+)\s*mW", re.MULTILINE),
    "package": re.compile(r"^Combined Power \(CPU \+ GPU \+ ANE\):\s+([\d.]+)\s*mW", re.MULTILINE),
}


class PowerMetricsSampler(BaseSampler):
    name = "powermetrics"

    def __init__(self) -> None:
        super().__init__()
        self._proc: subprocess.Popen | None = None
        self._out = None

    @classmethod
    def is_available(cls) -> bool:
        if sys.platform != "darwin":
            return False
        if shutil.which("powermetrics") is None:
            return False
        # `sudo -n` fails immediately rather than prompting. If the user has not
        # granted passwordless sudo we decline, so benchmarks stay unattended.
        try:
            return (
                subprocess.run(
                    ["sudo", "-n", "true"],
                    capture_output=True,
                    timeout=5,
                ).returncode
                == 0
            )
        except (subprocess.SubprocessError, OSError):
            return False

    def start(self) -> None:
        super().start()
        self._out = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+", suffix=".powermetrics", delete=False
        )
        self._proc = subprocess.Popen(
            [
                "sudo", "-n", "powermetrics",
                "--samplers", "cpu_power,gpu_power",
                "-i", str(_SAMPLE_INTERVAL_MS),
            ],
            stdout=self._out,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> PowerReading:
        duration = self._elapsed()
        if self._proc is None or self._out is None:
            return PowerReading(self.name, None, None, duration, note="sampler was never started")

        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)

        self._out.flush()
        try:
            with open(self._out.name) as fh:
                text = fh.read()
        except OSError:
            text = ""
        finally:
            self._cleanup()

        components: dict[str, float] = {}
        for key, pattern in _PATTERNS.items():
            values = [float(v) for v in pattern.findall(text)]
            if values:
                components[key] = sum(values) / len(values) / 1000.0  # mW -> W

        if not components:
            return PowerReading(
                self.name, None, None, duration,
                note="powermetrics produced no parseable samples",
            )

        # Prefer the package total; otherwise sum what we did capture.
        avg_w = components.get("package") or sum(
            v for k, v in components.items() if k in ("cpu", "gpu", "ane")
        )
        return PowerReading(
            sampler=self.name,
            avg_power_w=avg_w,
            energy_j=avg_w * duration,
            duration_s=duration,
            components_w=components,
        )

    def _cleanup(self) -> None:
        import contextlib  # noqa: PLC0415
        import os  # noqa: PLC0415

        if self._out is not None:
            with contextlib.suppress(OSError):
                self._out.close()
            with contextlib.suppress(OSError):
                os.unlink(self._out.name)
        self._out = None
        self._proc = None
