"""
PPO Profiler — writes Chrome Trace Event Format JSON for ui.perfetto.dev.

Usage:
    profiler = PPOProfiler()
    with profiler.span("rollout"):
        with profiler.span("rollout/policy_forward"):
            ...
    profiler.save("trace.json")

Or with no profiling (zero overhead):
    profiler = NullProfiler()
    with profiler.span("anything"):   # no-op context manager
        ...
"""

import json
import os
import time
from contextlib import contextmanager


class PPOProfiler:
    """Records begin/end spans and writes Perfetto-compatible trace JSON."""

    def __init__(self, pid=0, tid=0):
        self._events = []
        self._pid = pid
        self._tid = tid
        self._t0 = time.perf_counter()

    def _us(self):
        """Microseconds since profiler creation."""
        return (time.perf_counter() - self._t0) * 1e6

    @contextmanager
    def span(self, name: str):
        ts_begin = self._us()
        self._events.append({
            "ph": "B", "name": name,
            "ts": ts_begin, "pid": self._pid, "tid": self._tid,
        })
        try:
            yield
        finally:
            ts_end = self._us()
            self._events.append({
                "ph": "E", "name": name,
                "ts": ts_end, "pid": self._pid, "tid": self._tid,
            })

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"traceEvents": self._events}, f)
        print(f"Trace saved: {path}  ({len(self._events)} events)")

    def __len__(self):
        return len(self._events)


class NullProfiler:
    """Drop-in replacement that does nothing — zero overhead."""

    @contextmanager
    def span(self, name: str):
        yield

    def save(self, path: str):
        pass
