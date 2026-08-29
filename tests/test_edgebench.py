"""Tests for the measurement primitives.

Focused on the things that are easy to get subtly wrong and hard to notice:
percentile ordering, the TTFT/decode split, and result persistence.
"""

from __future__ import annotations

import edgebench
from edgebench import timing
from edgebench.store import Result, Store


def test_summarize_orders_percentiles():
    stats = timing.summarize([i / 1000 for i in range(1, 101)])
    assert stats.n == 100
    assert stats.min_ms <= stats.p50_ms <= stats.p95_ms <= stats.p99_ms <= stats.max_ms


def test_summarize_single_sample_has_zero_stdev():
    stats = timing.summarize([0.005])
    assert stats.p50_ms == stats.p99_ms == 5.0
    assert stats.stdev_ms == 0.0


def test_make_sync_is_callable_without_torch():
    """Callers never branch on platform, so this must work with no framework."""
    timing.make_sync("cpu")()


def test_generation_timer_splits_prefill_from_decode():
    """TTFT and throughput are different problems; averaging them hides the tradeoff."""
    gen = timing.GenerationTimer()
    gen.start()
    for _ in range(5):
        gen.token()
    stats = gen.finish()

    assert stats.ttft_ms is not None and stats.ttft_ms >= 0
    # The first token ends prefill and is not part of the decode measurement.
    assert stats.tokens_generated == 4


def test_generation_timer_without_tokens_is_empty_not_wrong():
    gen = timing.GenerationTimer()
    gen.start()
    stats = gen.finish()
    assert stats.tokens_generated == 0
    assert stats.tokens_per_s is None


def test_benchmark_populates_a_complete_row():
    result = edgebench.benchmark(
        lambda: sum(range(500)),
        label="unit-test",
        project="tests",
        backend="python",
        iters=5,
        warmup=2,
        quality=("dummy", 1.0),
        store=False,
    )
    assert result.lat_p50_ms > 0
    assert result.peak_rss_mb > 0
    assert result.power_sampler is not None  # "null" counts; None does not
    assert result.device in edgebench.device.available()
    assert result.quality_metric == "dummy"
    assert result.platform and result.python_version


def test_store_roundtrip(tmp_path):
    db_path = tmp_path / "t.db"
    with Store(db_path) as db:
        db.add(Result(label="a", project="p", lat_p50_ms=1.5, quality_value=0.9))
        db.add(Result(label="b", project="p", lat_p50_ms=2.5))
        assert db.count() == 2
        assert db.projects() == ["p"]
        rows = db.query(project="p")
        assert {r["label"] for r in rows} == {"a", "b"}


def test_store_preserves_extra_as_json(tmp_path):
    with Store(tmp_path / "t.db") as db:
        db.add(Result(label="x", extra={"power_components_w": {"cpu": 1.5}}))
        row = db.query(limit=1)[0]
        assert "power_components_w" in row["extra_json"]


def test_result_summary_omits_missing_fields():
    """Rows are sparse by design -- an LLM row has no accuracy."""
    summary = Result(label="only-a-label").summary()
    assert summary == "only-a-label"
