"""Best-effort host utilization sampling; no GPU Python dependency required."""

import os
import subprocess
from pathlib import Path
from threading import Event, Thread

RESOURCE_FIELDS = (
    "cpu_util_avg_pct",
    "cpu_util_max_pct",
    "gpu_util_avg_pct",
    "gpu_util_max_pct",
)


def _cpu_ticks() -> tuple[int, int] | None:
    try:
        values = [
            int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
        ]
        return sum(values), values[3] + values[4]
    except (OSError, ValueError, IndexError):
        return None


def _gpu_utilization() -> list[float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=0.5,
            check=True,
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        selected = None if visible is None else {x.strip() for x in visible.split(",")}
        values = []
        for row in result.stdout.splitlines():
            index, uuid, utilization = [x.strip() for x in row.split(",")]
            if (
                selected is None
                or index in selected
                or any(uuid.startswith(x) for x in selected if x.startswith("GPU-"))
            ):
                value = float(utilization)
                if 0 <= value <= 100:
                    values.append(value)
        return values
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


class ResourceSampler:
    """Sample Linux host CPU and selected physical GPU utilization every 0.5 s."""

    def __init__(self) -> None:
        self.cpu: list[float] = []
        self.gpu: list[float] = []
        self.previous = _cpu_ticks()
        self.stopped = Event()
        self.thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _sample_cpu(self) -> None:
        current = _cpu_ticks()
        if current is not None and self.previous is not None:
            total = current[0] - self.previous[0]
            idle = current[1] - self.previous[1]
            if total > 0:
                self.cpu.append(max(0, min(100, 100 * (total - idle) / total)))
        self.previous = current

    def _run(self) -> None:
        while not self.stopped.is_set():
            self.gpu.extend(_gpu_utilization())
            if self.stopped.wait(0.5):
                break
            self._sample_cpu()

    def finish(self) -> dict[str, float]:
        self.stopped.set()
        self.thread.join()
        self._sample_cpu()
        metrics = {}
        for name, values in [("cpu", self.cpu), ("gpu", self.gpu)]:
            if values:
                metrics[f"{name}_util_avg_pct"] = sum(values) / len(values)
                metrics[f"{name}_util_max_pct"] = max(values)
        return metrics
