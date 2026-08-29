"""NVIDIA GPU power sampling via NVML.

This is the sampler that will run when the curriculum's code is moved to a CUDA
box, which is the whole point of keeping power behind an interface. It cannot
run on this machine -- `is_available()` returns False and the registry falls
through to the next candidate.

Polls in a background thread because NVML reports instantaneous draw, so an
average over the window requires sampling it.
"""

from __future__ import annotations

import threading

from .base import BaseSampler, PowerReading

_POLL_INTERVAL_S = 0.1


class NvmlSampler(BaseSampler):
    name = "nvml"

    def __init__(self, device_index: int = 0) -> None:
        super().__init__()
        self._index = device_index
        self._samples_w: list[float] = []
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._handle = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            try:
                return pynvml.nvmlDeviceGetCount() > 0
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            # NVML raises a family of its own exception types; any of them means
            # "not usable here", which is not an error worth propagating.
            return False

    def start(self) -> None:
        super().start()
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._index)
        self._samples_w = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        import pynvml  # noqa: PLC0415

        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples_w.append(mw / 1000.0)
            except Exception:
                break
            self._stop_event.wait(_POLL_INTERVAL_S)

    def stop(self) -> PowerReading:
        duration = self._elapsed()
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlShutdown()
        except Exception:
            pass

        if not self._samples_w:
            return PowerReading(
                self.name, None, None, duration, note="NVML returned no samples"
            )

        avg_w = sum(self._samples_w) / len(self._samples_w)
        return PowerReading(
            sampler=self.name,
            avg_power_w=avg_w,
            energy_j=avg_w * duration,
            duration_s=duration,
            components_w={"gpu": avg_w},
            note=f"{len(self._samples_w)} samples @ {_POLL_INTERVAL_S}s",
        )
