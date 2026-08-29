"""Provenance capture.

A benchmark number is worthless six months later if you cannot say what produced
it. Since this curriculum deliberately compares results across phases -- a
pruned CNN from month one against a quantized LLM from month four -- every row
carries enough context to be re-derived or invalidated.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_HASH_CHUNK = 1024 * 1024
#: Hash only the head of very large files. A GGUF can be gigabytes, and reading
#: it in full on every run would dominate the benchmark it is meant to describe.
_HASH_LIMIT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Provenance:
    git_sha: str | None
    git_dirty: bool
    platform: str
    python_version: str
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def git_state(repo: Path | None = None) -> tuple[str | None, bool]:
    """Return (short SHA, dirty). (None, False) when not in a git repo."""
    cwd = str(repo or Path(__file__).resolve().parent.parent)

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "--short", "HEAD")
    if sha is None:
        return None, False

    status = _git("status", "--porcelain")
    return sha, bool(status)


def capture(**extra) -> Provenance:
    """Snapshot the environment for a benchmark row."""
    sha, dirty = git_state()
    return Provenance(
        git_sha=sha,
        git_dirty=dirty,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=sys.version.split()[0],
        extra=extra,
    )


def hash_path(path: str | Path) -> tuple[str | None, int | None]:
    """Return (sha256 hex, total bytes) for a model file or directory.

    Directories hash the sorted relative paths alongside their contents, so a
    renamed shard changes the hash. Returns (None, None) for a path that does
    not exist -- a HuggingFace repo id, for instance, which is recorded as a
    plain `model_ref` instead.
    """
    p = Path(path)
    if not p.exists():
        return None, None

    digest = hashlib.sha256()
    total = 0

    files = sorted(p.rglob("*")) if p.is_dir() else [p]
    for f in files:
        if not f.is_file():
            continue
        if p.is_dir():
            digest.update(str(f.relative_to(p)).encode())
        size = f.stat().st_size
        total += size
        remaining = _HASH_LIMIT_BYTES
        with open(f, "rb") as fh:
            while remaining > 0:
                chunk = fh.read(min(_HASH_CHUNK, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)

    return digest.hexdigest(), total


def backend_version(backend: str) -> str | None:
    """Version string for a named backend, best effort."""
    modules = {
        "torch": "torch",
        "onnx": "onnxruntime",
        "onnxruntime": "onnxruntime",
        "coreml": "coremltools",
        "mlx": "mlx",
        "transformers": "transformers",
    }
    module_name = modules.get(backend.lower())
    if module_name is None:
        return None
    try:
        import importlib  # noqa: PLC0415

        return getattr(importlib.import_module(module_name), "__version__", None)
    except ImportError:
        return None
