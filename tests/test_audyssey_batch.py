import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from audyssey.cli import app
from audyssey.settings import Settings
from audyssey.media import MediaError
from tests.audyssey_media import make_video


class BatchTests(unittest.TestCase):
    def test_batch_renders_and_records_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = make_video(root / "clip.mp4")
            settings = Settings(
                work_dir=root / "jobs",
                planner="stub",
                separator="stub",
                filter="stub",
                grounder="stub",
                generator="stub",
            )
            with patch("audyssey.cli.Settings", return_value=settings):
                result = CliRunner().invoke(app, ["batch", str(video)])
            self.assertEqual(result.exit_code, 0, result.output)
            rows = list(
                csv.DictReader(
                    io.StringIO((settings.work_dir / "batch.csv").read_text())
                )
            )
            self.assertEqual(rows[0]["status"], "done")
            directory = settings.work_dir / rows[0]["job_id"]
            self.assertTrue((directory / "out/video.mp4").is_file())
            self.assertTrue((settings.work_dir / "timings.csv").is_file())
            self.assertGreater(float(rows[0]["duration_s"]), 0)

    def test_batch_continues_after_invalid_video_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.mp4"
            bad.write_bytes(b"bad")
            settings = Settings(
                work_dir=root / "jobs",
                planner="stub",
                separator="stub",
                filter="stub",
                grounder="stub",
                generator="stub",
            )
            with (
                patch("audyssey.cli.Settings", return_value=settings),
                patch(
                    "audyssey.batch.check_upload",
                    side_effect=[MediaError("invalid"), MediaError("second")],
                ) as check,
            ):
                result = CliRunner().invoke(app, ["batch", str(bad), str(bad)])
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertEqual(check.call_count, 2)
            rows = list(
                csv.DictReader(
                    io.StringIO((settings.work_dir / "batch.csv").read_text())
                )
            )
            self.assertEqual([r["status"] for r in rows], ["failed", "failed"])
