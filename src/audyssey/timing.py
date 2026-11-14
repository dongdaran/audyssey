"""Small context-manager based timing and CSV recording helpers."""

import csv
import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterator, Mapping

_LOGGER = logging.getLogger(__name__)
_CSV_LOCK = Lock()
_ACTIVE_SAMPLE: contextvars.ContextVar["TimingSample | None"] = contextvars.ContextVar(
    "audyssey_timing_sample", default=None
)
_TOOL_METRICS: contextvars.ContextVar[dict[str, float] | None] = (
    contextvars.ContextVar("audyssey_tool_timing_metrics", default=None)
)

_CSV_FIELDS = (
    "timestamp_utc",
    "job_id",
    "stage",
    "source_id",
    "round",
    "attempt",
    "status",
    "duration_s",
    "model_load_s",
    "processing_s",
    "model_id",
)


@dataclass
class TimingSample:
    path: Path
    job_id: str
    stage: str
    source_id: str | None = None
    round_number: int | None = None
    attempt: int | None = None
    model_id: str | None = None
    status: str = "done"
    duration_s: float = 0.0
    model_load_s: float | None = None
    processing_s: float | None = None
    timestamp_utc: str = ""


@contextmanager
def measure(
    path: Path,
    *,
    job_id: str,
    stage: str,
    source_id: str | None = None,
    round_number: int | None = None,
    attempt: int | None = None,
    model_id: str | None = None,
) -> Iterator[TimingSample]:
    """Measure one pipeline operation and append one CSV row on exit."""
    sample = TimingSample(
        path=path,
        job_id=job_id,
        stage=stage,
        source_id=source_id,
        round_number=round_number,
        attempt=attempt,
        model_id=model_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    token = _ACTIVE_SAMPLE.set(sample)
    started = time.perf_counter()
    try:
        yield sample
    except Exception:
        sample.status = "failed"
        raise
    finally:
        sample.duration_s = time.perf_counter() - started
        _ACTIVE_SAMPLE.reset(token)
        _append_sample(sample)


@contextmanager
def measure_tool_metric(name: str) -> Iterator[None]:
    """Measure a model-load or processing span inside a tool request."""
    started = time.perf_counter()
    try:
        yield
    finally:
        duration_s = time.perf_counter() - started
        sample = _ACTIVE_SAMPLE.get()
        if sample is not None:
            if name == "model_load":
                sample.model_load_s = (sample.model_load_s or 0.0) + duration_s
            elif name == "processing":
                sample.processing_s = (sample.processing_s or 0.0) + duration_s
        metrics = _TOOL_METRICS.get()
        if metrics is not None:
            metrics[name] = metrics.get(name, 0.0) + duration_s


@contextmanager
def collect_tool_metrics() -> Iterator[dict[str, float]]:
    """Collect nested tool timings for one HTTP request."""
    metrics: dict[str, float] = {}
    token = _TOOL_METRICS.set(metrics)
    try:
        yield metrics
    finally:
        _TOOL_METRICS.reset(token)


def server_timing_header(metrics: Mapping[str, float]) -> str:
    """Format timings in milliseconds using the Server-Timing header syntax."""
    return ", ".join(
        f"{name};dur={seconds * 1000:.3f}"
        for name, seconds in metrics.items()
        if name in {"model_load", "processing"}
    )


def record_server_timing(header: str | None) -> None:
    """Copy Server-Timing metrics from a remote tool response into its caller."""
    sample = _ACTIVE_SAMPLE.get()
    if sample is None or not header:
        return
    for entry in header.split(","):
        name, *parameters = entry.strip().split(";")
        duration_ms = next(
            (
                parameter.partition("=")[2]
                for parameter in parameters
                if parameter.strip().startswith("dur=")
            ),
            None,
        )
        try:
            seconds = float(duration_ms) / 1000 if duration_ms is not None else None
        except ValueError:
            continue
        if seconds is None or seconds < 0:
            continue
        if name == "model_load":
            sample.model_load_s = (sample.model_load_s or 0.0) + seconds
        elif name == "processing":
            sample.processing_s = (sample.processing_s or 0.0) + seconds


def _append_sample(sample: TimingSample) -> None:
    row = {
        "timestamp_utc": sample.timestamp_utc,
        "job_id": sample.job_id,
        "stage": sample.stage,
        "source_id": sample.source_id or "",
        "round": "" if sample.round_number is None else sample.round_number,
        "attempt": "" if sample.attempt is None else sample.attempt,
        "status": sample.status,
        "duration_s": f"{sample.duration_s:.6f}",
        "model_load_s": (
            "" if sample.model_load_s is None else f"{sample.model_load_s:.6f}"
        ),
        "processing_s": (
            "" if sample.processing_s is None else f"{sample.processing_s:.6f}"
        ),
        "model_id": sample.model_id or "",
    }
    try:
        sample.path.parent.mkdir(parents=True, exist_ok=True)
        with _CSV_LOCK:
            needs_header = not sample.path.exists() or sample.path.stat().st_size == 0
            with sample.path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
    except OSError:
        _LOGGER.exception("could not append timing row to %s", sample.path)
