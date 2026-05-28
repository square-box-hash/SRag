# srag/tracer.py
"""
SRagTracer — per-step timing and debug tracing.
Zero overhead when tracing is disabled.
"""
from __future__ import annotations
import time
import logging
from contextlib import contextmanager
from srag.result import SRagTrace

logger = logging.getLogger(__name__)


class SRagTracer:
    """
    Lightweight per-operation tracer.
    Tracks timing for each pipeline step.
    
    Usage:
        tracer = SRagTracer(enabled=True)
        with tracer.step("fetch"):
            docs = await scraper.get_facts(query)
        print(tracer.build().summary())
    """

    def __init__(self, enabled: bool = False):
        self.enabled  = enabled
        self._steps:  dict[str, float] = {}
        self._start:  dict[str, float] = {}
        self._total_start = time.perf_counter()

    @contextmanager
    def step(self, name: str):
        """Context manager for timing a pipeline step."""
        if not self.enabled:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms       = (time.perf_counter() - t0) * 1000
            self._steps[name] = elapsed_ms
            logger.debug("[trace] %s: %.1fms", name, elapsed_ms)

    def log(self, message: str) -> None:
        if self.enabled:
            logger.debug("[trace] %s", message)

    def build(
        self,
        doc_count:   int = 0,
        chunk_count: int = 0,
        topic:       str = "",
        mode:        str = "simple",
    ) -> SRagTrace:
        """Build a SRagTrace from recorded steps."""
        total_ms = (time.perf_counter() - self._total_start) * 1000
        return SRagTrace(
            fetch_ms    = self._steps.get("fetch",  0.0),
            chunk_ms    = self._steps.get("chunk",  0.0),
            embed_ms    = self._steps.get("embed",  0.0),
            rerank_ms   = self._steps.get("rerank", 0.0),
            total_ms    = total_ms,
            doc_count   = doc_count,
            chunk_count = chunk_count,
            topic       = topic,
            mode        = mode,
        )

    def print_summary(self) -> None:
        """Print timing breakdown to stdout."""
        if not self.enabled or not self._steps:
            return
        total_ms = (time.perf_counter() - self._total_start) * 1000
        print(f"\n  [trace] Pipeline timing:")
        for step, ms in self._steps.items():
            bar = "█" * int(ms / total_ms * 20) if total_ms > 0 else ""
            print(f"  {step:<12} {ms:>7.1f}ms  {bar}")
        print(f"  {'total':<12} {total_ms:>7.1f}ms")