"""Read-only browser for batch results; no inference initialization."""
import json
from pathlib import Path
from functools import lru_cache
from importlib.resources import files

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from audyssey.media import read_wav


def create_results_app(root: Path) -> FastAPI:
    root = root.resolve()
    app = FastAPI()

    def resolve(rel: str) -> Path:
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, 'File unavailable')
        return path

    @app.get('/')
    def index():
        return HTMLResponse(files('audyssey.devui').joinpath('results.html').read_text())

    @app.get('/api/results')
    def listing():
        import csv
        runs = []
        for run in sorted(root.glob('run0*'), reverse=True):
            if not run.is_dir() or not run.resolve().is_relative_to(root):
                continue
            names = {}
            try:
                with (run / 'batch.csv').open() as stream:
                    names = {r['job_id']: Path(r['input']).name for r in csv.DictReader(stream)}
            except (OSError, KeyError):
                pass
            jobs = []
            for path in sorted(run.glob('*/job.json')):
                try:
                    job = json.loads(resolve(str(path.relative_to(root))).read_text())
                    jobs.append(dict(key=str(path.relative_to(root)), name=names.get(path.parent.name, path.parent.name), job=job))
                except (ValueError, OSError, HTTPException) as exc:
                    jobs.append(dict(name=path.parent.name, error=str(exc)))
            runs.append(dict(name=run.name, jobs=jobs))
        return runs

    @app.get('/api/file')
    def media(path: str):
        return FileResponse(resolve(path))

    @lru_cache(maxsize=256)
    def peaks(path: str, size: int, modified: int):
        data = read_wav(Path(path))
        data = np.abs(data).reshape(len(data), -1).max(axis=1)
        return [float(np.max(chunk)) if len(chunk) else 0 for chunk in np.array_split(data, min(1800, max(1, len(data))))]

    @app.get('/api/peaks')
    def waveform(path: str):
        p = resolve(path)
        try:
            stat = p.stat()
            return peaks(str(p), stat.st_size, stat.st_mtime_ns)
        except Exception as exc:
            raise HTTPException(422, 'Waveform unavailable') from exc

    return app
