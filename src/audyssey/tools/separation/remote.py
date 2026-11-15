from io import BytesIO
from zipfile import ZipFile
from pathlib import Path

from audyssey.contracts import SourcePlan
from audyssey.settings import Settings
from audyssey.tools.remote import post


def separate(
    mono: Path, plan: SourcePlan, video: Path | None, out: Path, attempt: int
) -> Path:
    settings = Settings()
    files = {"mono": ("mono.wav", mono.read_bytes(), "audio/wav")}
    if video is not None:
        files["video"] = ("video.mp4", video.read_bytes(), "video/mp4")
    response = post(
        settings.separator_url,
        "/separate",
        files,
        {"plan": plan.model_dump_json(), "attempt": str(attempt), "include_residual": "true"},
        settings.tool_token,
        settings.remote_timeout_s,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if response.headers.get("content-type", "").split(";")[0] == "application/zip":
        with ZipFile(BytesIO(response.content)) as archive:
            out.write_bytes(archive.read("track.wav"))
            out.with_name(f"{out.stem}_resi.wav").write_bytes(archive.read("track_resi.wav"))
    else:
        out.write_bytes(response.content)
    return out
