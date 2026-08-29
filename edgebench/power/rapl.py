"""Linux CPU package energy via Intel RAPL.

Reads the powercap sysfs energy counters. Unlike NVML this is a true energy
counter rather than an instantaneous reading, so the measurement is just a
difference across the window -- no polling thread, no sampling error.

The counters wrap, hence the `max_energy_range_uj` correction.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .base import BaseSampler, PowerReading

_RAPL_ROOT = Path("/sys/class/powercap")


def _domains() -> list[Path]:
    """Top-level RAPL package domains (skip the sub-domains to avoid double counting)."""
    if not _RAPL_ROOT.is_dir():
        return []
    return sorted(
        d for d in _RAPL_ROOT.glob("intel-rapl:*")
        if d.is_dir() and (d / "energy_uj").exists() and ":" not in d.name.split("intel-rapl:")[1]
    )


def _read_uj(domain: Path) -> int | None:
    try:
        return int((domain / "energy_uj").read_text().strip())
    except (OSError, ValueError):
        return None


def _max_uj(domain: Path) -> int | None:
    try:
        return int((domain / "max_energy_range_uj").read_text().strip())
    except (OSError, ValueError):
        return None


class RaplSampler(BaseSampler):
    name = "rapl"

    def __init__(self) -> None:
        super().__init__()
        self._start_uj: dict[Path, int] = {}

    @classmethod
    def is_available(cls) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        # Readability matters as much as existence: since Linux 5.10 these
        # counters are commonly root-only as a side-channel mitigation.
        return any(_read_uj(d) is not None for d in _domains())

    def start(self) -> None:
        super().start()
        self._start_uj = {}
        for domain in _domains():
            value = _read_uj(domain)
            if value is not None:
                self._start_uj[domain] = value

    def stop(self) -> PowerReading:
        duration = self._elapsed()
        if not self._start_uj:
            return PowerReading(self.name, None, None, duration, note="sampler was never started")

        total_uj = 0
        components: dict[str, float] = {}
        for domain, start_value in self._start_uj.items():
            end_value = _read_uj(domain)
            if end_value is None:
                continue
            delta = end_value - start_value
            if delta < 0:  # counter wrapped
                ceiling = _max_uj(domain)
                if ceiling is None:
                    continue
                delta += ceiling
            total_uj += delta
            if duration > 0:
                label = (domain / "name").read_text().strip() if (domain / "name").exists() else domain.name
                components[label] = delta / 1e6 / duration

        if total_uj == 0 or duration <= 0:
            return PowerReading(self.name, None, None, duration, note="no RAPL energy delta")

        energy_j = total_uj / 1e6
        return PowerReading(
            sampler=self.name,
            avg_power_w=energy_j / duration,
            energy_j=energy_j,
            duration_s=duration,
            components_w=components,
        )
