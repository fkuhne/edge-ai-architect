"""SQLite results store.

One table, wide and flat. That is a deliberate choice: the point of this
database is that a single query can put a pruned CNN and a 4-bit LLM on the same
axes, and joins across a normalised schema would make the ad-hoc comparisons
this curriculum runs on far more tedious than they need to be.

Nullable columns are the norm -- an LLM row has no `quality_value` for accuracy,
a vision row has no `tokens_per_s` -- and that is fine.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "results" / "bench.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT    NOT NULL,
    label             TEXT    NOT NULL,
    project           TEXT,

    -- what ran
    model_ref         TEXT,
    model_hash        TEXT,
    model_bytes       INTEGER,
    quantization      TEXT,
    backend           TEXT,
    backend_version   TEXT,
    device            TEXT,
    device_name       TEXT,

    -- latency
    batch_size        INTEGER,
    iters             INTEGER,
    warmup            INTEGER,
    lat_mean_ms       REAL,
    lat_p50_ms        REAL,
    lat_p95_ms        REAL,
    lat_p99_ms        REAL,
    lat_stdev_ms      REAL,
    ttft_ms           REAL,
    tokens_per_s      REAL,

    -- memory
    peak_rss_mb       REAL,
    accel_peak_mb     REAL,

    -- power
    power_sampler     TEXT,
    avg_power_w       REAL,
    energy_j          REAL,

    -- quality (never report speed without it)
    quality_metric    TEXT,
    quality_value     REAL,

    -- provenance
    git_sha           TEXT,
    git_dirty         INTEGER,
    platform          TEXT,
    python_version    TEXT,

    notes             TEXT,
    extra_json        TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
CREATE INDEX IF NOT EXISTS idx_runs_label   ON runs(label);
CREATE INDEX IF NOT EXISTS idx_runs_ts      ON runs(ts);
"""


@dataclass
class Result:
    """One benchmark row."""

    label: str
    project: str | None = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    model_ref: str | None = None
    model_hash: str | None = None
    model_bytes: int | None = None
    quantization: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    device: str | None = None
    device_name: str | None = None

    batch_size: int | None = None
    iters: int | None = None
    warmup: int | None = None
    lat_mean_ms: float | None = None
    lat_p50_ms: float | None = None
    lat_p95_ms: float | None = None
    lat_p99_ms: float | None = None
    lat_stdev_ms: float | None = None
    ttft_ms: float | None = None
    tokens_per_s: float | None = None

    peak_rss_mb: float | None = None
    accel_peak_mb: float | None = None

    power_sampler: str | None = None
    avg_power_w: float | None = None
    energy_j: float | None = None

    quality_metric: str | None = None
    quality_value: float | None = None

    git_sha: str | None = None
    git_dirty: bool = False
    platform: str | None = None
    python_version: str | None = None

    notes: str | None = None
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = asdict(self)
        row["extra_json"] = json.dumps(row.pop("extra") or {})
        row["git_dirty"] = int(row["git_dirty"])
        return row

    def summary(self) -> str:
        parts = [f"{self.label}"]
        if self.backend:
            parts.append(f"{self.backend}/{self.device}")
        if self.lat_p50_ms is not None:
            parts.append(f"p50={self.lat_p50_ms:.2f}ms")
        if self.tokens_per_s is not None:
            parts.append(f"{self.tokens_per_s:.1f} tok/s")
        if self.peak_rss_mb is not None:
            parts.append(f"rss={self.peak_rss_mb:.0f}MB")
        if self.avg_power_w is not None:
            parts.append(f"{self.avg_power_w:.1f}W")
        if self.quality_value is not None:
            parts.append(f"{self.quality_metric}={self.quality_value:.4g}")
        return "  ".join(parts)


class Store:
    """Thin SQLite wrapper. Safe to construct repeatedly; schema is idempotent."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def add(self, result: Result) -> int:
        row = result.as_row()
        columns = [f.name for f in fields(Result) if f.name != "extra"] + ["extra_json"]
        placeholders = ", ".join(f":{c}" for c in columns)
        cur = self._conn.execute(
            f"INSERT INTO runs ({', '.join(columns)}) VALUES ({placeholders})", row
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def query(
        self,
        project: str | None = None,
        label: str | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM runs"
        clauses, params = [], {}
        if project:
            clauses.append("project = :project")
            params["project"] = project
        if label:
            clauses.append("label LIKE :label")
            params["label"] = label
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return list(self._conn.execute(sql, params))

    def projects(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT project FROM runs WHERE project IS NOT NULL ORDER BY project"
        )
        return [r["project"] for r in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
