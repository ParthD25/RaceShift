"""Wall-clock and memory measurement for training runs.

Forward-Forward's practical argument is lower peak training memory, so every run records
it. Peak RSS is the process high-water mark (includes data loading); the traced peak is
Python/NumPy allocations made inside the measured block only.
"""
from __future__ import annotations

import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ResourceReport:
    wall_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    peak_traced_mb: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out = {
            "wall_seconds": round(self.wall_seconds, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_traced_mb": round(self.peak_traced_mb, 1),
        }
        out.update({k: round(v, 4) for k, v in self.extras.items()})
        return out


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return usage / 1024.0 if sys.platform != "darwin" else usage / (1024.0 * 1024.0)


@contextmanager
def measure(report: ResourceReport | None = None):
    """Measure wall time and peak memory for the enclosed block."""
    report = report if report is not None else ResourceReport()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    start = time.perf_counter()
    try:
        yield report
    finally:
        report.wall_seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        report.peak_traced_mb = peak / (1024.0 * 1024.0)
        report.peak_rss_mb = _rss_mb()
        if not was_tracing:
            tracemalloc.stop()


def directory_bytes(path) -> int:
    from pathlib import Path

    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
