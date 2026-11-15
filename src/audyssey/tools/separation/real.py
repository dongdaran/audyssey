from functools import cache
from importlib import import_module
import math
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from threading import Lock

import numpy as np

from audyssey.contracts import SourcePlan
from audyssey.media import read_wav, write_wav
from audyssey.settings import Settings
from audyssey.timing import measure_tool_metric

# shortcut: one GPU model per process, one call at a time; batch or queue if throughput matters
_model_lock = Lock()


@cache
def _load():
    settings = Settings()
    try:
        torch = import_module("torch")
        sam = import_module("sam_audio")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM Audio is not installed: install the gpu/ environment"
        ) from exc
    # Rankers and the span predictor are ~4.4B of small's 5.8B params and are built in
    # SAMAudio.__init__ (the stock Judge ranker loads facebook/sam-audio-judge). Skip or
    # rebuild them via config overrides; they are not in the checkpoint, so loading
    # stays strict.
    overrides: dict[str, object] = {}
    if settings.sam_audio_reranking_candidates == 1:
        overrides.update(visual_ranker=None, text_ranker=None)
    else:
        # Same ensemble as the released configs, but the Judge ranker uses our
        # configured Judge (official or mirror) instead of the hardcoded official repo.
        clap = settings.sam_audio_clap_checkpoint or None
        imagebind = settings.sam_audio_imagebind_checkpoint
        overrides["text_ranker"] = {
            "kind": "ensemble",
            "rankers": {
                "clap": [{"kind": "clap", "checkpoint": clap}, 5.0],
                "judge": [
                    {
                        "kind": "judge",
                        "checkpoint_or_model_id": settings.sam_audio_judge_model,
                    },
                    1.0,
                ],
            },
        }
        overrides["visual_ranker"] = (
            {"kind": "imagebind", "checkpoint": imagebind} if imagebind else None
        )
    if not settings.sam_audio_predict_spans:
        overrides["span_predictor"] = None
    model = sam.SAMAudio.from_pretrained(settings.sam_audio_model, **overrides)
    model = model.to(settings.sam_audio_device).eval()
    processor = sam.SAMAudioProcessor.from_pretrained(settings.sam_audio_model)
    return torch, model, processor


def separate(
    mono: Path, plan: SourcePlan, video: Path | None, out: Path, attempt: int
) -> Path:
    settings = Settings()
    with _model_lock:
        with measure_tool_metric("model_load"):
            torch, model, processor = _load()
        with measure_tool_metric("processing"):
            audio = read_wav(mono)
            track = np.zeros((len(audio), 1), dtype=np.float32)
            weights = np.zeros((len(audio), 1), dtype=np.float32)
            # Each event gets its own context, including half a second on both sides.
            windows = []
            for span in plan.spans:
                if span.start >= len(audio) / 48000 or span.end <= 0:
                    continue
                start = max(0, min(len(audio), math.floor((span.start - 0.5) * 48000)))
                end = max(0, min(len(audio), math.ceil((span.end + 0.5) * 48000)))
                if start < end:
                    windows.append((start, end, span))
            if not plan.spans:
                windows = [(0, len(audio), None)]
            if not windows:
                raise ValueError("no planning spans overlap the input audio")
            with TemporaryDirectory() as tmp:
                for index, (start, end, span) in enumerate(windows):
                    offset = start / 48000
                    anchors = (
                        [[["+", max(0., span.start - offset),
                           min((end - start) / 48000, span.end - offset)]]]
                        if span is not None else None
                    )
                    cropped_video = video
                    if video is not None and span is not None:
                        cropped_video = Path(tmp) / f"{index}.mp4"
                        subprocess.run([
                            "ffmpeg", "-v", "error", "-y", "-i", str(video),
                            "-ss", str(offset), "-t", str((end - start) / 48000),
                            "-an", "-c:v", "libx264", str(cropped_video),
                        ], check=True, capture_output=True, text=True)
                    batch = processor(
                        descriptions=[plan.name],
                        audios=[torch.from_numpy(audio[start:end].T)],
                        anchors=anchors,
                        **({"masked_videos": [str(cropped_video)]}
                           if cropped_video is not None else {}),
                    ).to(settings.sam_audio_device)
                    with torch.inference_mode():
                        torch.manual_seed(attempt)
                        result = model.separate(
                            batch,
                            reranking_candidates=settings.sam_audio_reranking_candidates,
                            predict_spans=settings.sam_audio_predict_spans,
                        )
                    target = result.target[0].detach().cpu().float().numpy().reshape(-1, 1)
                    length = min(end - start, len(target))
                    track[start:start + length] += target[:length]
                    weights[start:start + length] += 1
            # Overlapping context must not double the output amplitude.
            np.divide(track, weights, out=track, where=weights > 0)
            return write_wav(out, track)
