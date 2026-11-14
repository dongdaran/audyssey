#!/usr/bin/env bash
# Run the pipeline on RunPod; arguments are passed to `audyssey batch`.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode=real
if [[ "${1:-}" == "--stub" ]]; then
    mode=stub
    shift
fi
if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: bash script/run_test.sh [--stub] VIDEO [VIDEO ...] [--no-render]

Default: real GPU models in gpu/.venv, external planner configured via environment.
--stub: CPU smoke run with fake models in .venv; no API key or weights needed.
Set AUDYSSEY_WORK_DIR for output (default: work/jobs under the repository).
Requires uv, ffmpeg and ffprobe. Set HF_HOME for persistent model caching.
HELP
    if [[ $# -eq 0 ]]; then exit 2; fi
    exit 0
fi
for executable in uv ffmpeg ffprobe; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "Missing executable: $executable" >&2
        if [[ "$executable" != uv ]]; then
            echo 'On a RunPod root shell, install it with: apt-get update && apt-get install -y ffmpeg' >&2
        fi
        exit 2
    fi
done
export AUDYSSEY_WORK_DIR="${AUDYSSEY_WORK_DIR:-$repo_dir/work/jobs}"
if [[ "$mode" == stub ]]; then
    export AUDYSSEY_PLANNER=stub AUDYSSEY_SEPARATOR=stub AUDYSSEY_FILTER=stub
    export AUDYSSEY_GROUNDER=stub AUDYSSEY_GENERATOR=stub AUDYSSEY_VISUAL_PROMPT=false
    uv sync --project "$repo_dir" --locked --extra ui
    exec uv run --project "$repo_dir" --locked --extra ui audyssey batch "$@"
fi
# The same command can be run directly from the repository root.
cd "$repo_dir"
exec uv run --project gpu --env-file .env.runpod audyssey batch "$@"
