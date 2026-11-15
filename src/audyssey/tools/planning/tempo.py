"""TEMPO caption planner, executed in an isolated GPU environment."""
import json
import re
import subprocess
from pathlib import Path

from audyssey.contracts import PlanIssue, SourcePlan, Span
from audyssey.settings import Settings

_TIMESTAMP = r'<\|(\d+(?:\.\d+)?)\|>'
_EVENT = re.compile(_TIMESTAMP + r'\s*(.*?)\s*' + _TIMESTAMP, re.S)


def parse_captions(raw: str, duration: float) -> list[SourcePlan]:
    sources = []
    names: dict[str, int] = {}
    for start, caption, end in _EVENT.findall(raw):
        start, end = max(0., float(start)), min(duration, float(end))
        caption = caption.strip()
        if not caption or start >= end:
            continue
        kind = 'sfx'
        tag = re.match(r'^\[(speech|dialogue|music|sfx|background|ambience)\]\s*', caption, re.I)
        if tag:
            kind = {'speech': 'dialogue', 'background': 'ambience'}.get(tag[1].lower(), tag[1].lower())
            caption = caption[tag.end():].strip()
        if not caption:
            continue
        # Preserve distinct events: identical captions do not prove source identity.
        names[caption] = names.get(caption, 0) + 1
        name = caption if names[caption] == 1 else f'{caption} ({names[caption]})'
        sources.append(SourcePlan(name=name, kind=kind, description=caption,
                                  spans=[Span(start=start, end=end)], visual_hint=None))
    if not sources:
        raise RuntimeError('TEMPO returned no valid timestamped captions; inspect tempo-caption.json')
    return sources


def plan(video: Path, audio: Path, duration: float, feedback: list[PlanIssue]) -> list[SourcePlan]:
    settings = Settings()
    if duration > 60:
        raise ValueError('TEMPO currently supports clips up to 60 seconds')
    if not settings.tempo_python or not settings.tempo_repo:
        raise ValueError('Set AUDYSSEY_TEMPO_PYTHON and AUDYSSEY_TEMPO_REPO; see docs/tempo.md')
    runner = Path(__file__).resolve().parents[4] / 'experiments/tempo/infer.py'
    output = audio.parent / 'tempo-caption.json'
    subprocess.run([settings.tempo_python, str(runner), str(audio.resolve()),
                    '--repo', settings.tempo_repo, '--model', settings.tempo_model,
                    '--output', str(output.resolve())], check=True)
    return parse_captions(json.loads(output.read_text())['raw'], duration)


def validate_plan(video: Path, audio: Path, plans: list[SourcePlan]) -> list[PlanIssue]:
    # Keep the configured audio-capable LLM validator as a separate check.
    from audyssey.tools.planning.real import validate_plan as validate
    return validate(video, audio, plans)
