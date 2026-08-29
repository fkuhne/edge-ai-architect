"""The fallback sampler: always available, measures nothing.

This exists so that "no power measurement on this platform" is a normal,
recorded outcome rather than an exception. A benchmark row that says
`power_sampler = "null"` is honest; one that crashed is useless.
"""

from __future__ import annotations

from .base import BaseSampler, PowerReading


class NullSampler(BaseSampler):
    name = "null"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def stop(self) -> PowerReading:
        return PowerReading(
            sampler=self.name,
            avg_power_w=None,
            energy_j=None,
            duration_s=self._elapsed(),
            note="no power sampler available on this platform",
        )
