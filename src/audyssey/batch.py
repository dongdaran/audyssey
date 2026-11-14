"""Sequential file processing without an HTTP server."""

import csv
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal

import typer

from audyssey.contracts import Render
from audyssey.graph import Context, analyze, handle_command
from audyssey.media import check_upload
from audyssey.settings import Settings
from audyssey.store import JobStore
from audyssey.tools import load_tools


def run_batch(
    videos: Sequence[Path],
    settings: Settings,
    render: bool,
    mode: Literal["separate", "generate"],
) -> bool:
    ctx = Context(load_tools(settings), settings, JobStore(settings.work_dir))
    summary = ctx.store.root / "batch.csv"
    failed = False
    for video in videos:
        started = perf_counter()
        job_id, error, status = "", "", "done"
        try:
            duration = check_upload(video, settings.max_duration_s)
            job, directory = ctx.store.create(video.name, video.read_bytes())
            job_id = job.id
            job.duration = duration
            ctx.store.save(job)
            # 하나씩 진행
            job = analyze(job, ctx, mode=mode)
            if job.state == "failed":
                raise RuntimeError(job.error or "analysis failed")
            if render:
                handle_command(job, Render(include_residual_stem=True), ctx)
            if any(step.status == "warning" for step in job.steps):
                status = "warning"
            typer.echo(f"{video}: {status} → {directory}")
        except Exception as exc:
            failed = True
            status, error = "failed", str(exc)
            typer.echo(f"{video}: failed: {error}", err=True)
        row = dict(
            input=str(video.resolve()),
            job_id=job_id,
            status=status,
            duration_s=f"{perf_counter() - started:.6f}",
            error=error,
        )
        with summary.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
    typer.echo(f"Summary: {summary}")
    typer.echo(f"Stage timings: {ctx.store.root / 'timings.csv'}")
    return failed
