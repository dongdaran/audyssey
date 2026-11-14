# RunPod에서 CLI 배치 실행

RunPod PyTorch Pod에서 저장소를 clone하고 설정 파일을 한 번 작성하면, `uv run` 한 번으로 배치를 실행합니다.
웹 UI, main 서버, 도구 서버, 포트 공개는 필요하지 않습니다. GPU 모델은 CLI 프로세스 안에서 실행하고,
플래닝과 계획 검증은 설정한 OpenAI 호환 API를 호출합니다.

## 처음 한 번 준비

CUDA 12.8을 지원하는 NVIDIA 드라이버, uv, git, ffmpeg/ffprobe가 필요합니다.
RunPod 기본 torch 대신 `gpu/.venv`의 프로젝트 의존성을 사용합니다.
필요한 운영체제 프로그램이 없는 Pod에서는 root 셸에서 설치합니다.

```bash
apt-get update
apt-get install -y ffmpeg git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /workspace
git clone https://github.com/capstone-audissey-fall-2026/audyssey.git
cd audyssey
cp .env.runpod.example .env.runpod
```

`batch`가 포함된 브랜치/커밋을 사용해야 합니다. 로컬 수정은 push 전까지 clone에 포함되지 않습니다.
`.env.runpod`에서 다음 값을 채웁니다.

- `AUDYSSEY_PLANNER_API_KEY`: 플래너 API 키
- `HF_TOKEN`: [접근 권한](running.md#실제-모델로-실행하기-팀원용)을 받은 Hugging Face 읽기 토큰
- `HF_HOME`: 가중치를 보존할 볼륨 경로
- `AUDYSSEY_WORK_DIR`: 결과를 보존할 볼륨 경로

`.env.runpod`는 git에서 제외합니다. 생성용 비공개 모델 접근 권한도 필요합니다.
Gemini 모델 이름은 계정에서 사용 가능한 모델로 바꿀 수 있습니다.
이미 모든 가중치가 캐시에 준비돼 있다면 `HF_HUB_OFFLINE=1`로 설정할 수 있습니다.
SAM3 시각 힌트를 켜는 경우 필요한 가중치를 사전에 준비해야 합니다([가중치 준비](running.md)).

## 실행: 저장소 루트에서 명령 하나

```bash
uv run --project gpu --env-file .env.runpod audyssey batch /workspace/inputs/*.mp4
```

`uv run`이 Python 3.12 환경과 GPU·플래너 의존성을 준비하고 실행합니다. 별도 `uv sync`, `export`, `--with`는 필요하지 않습니다.
`pyproject.toml`은 Python 의존성을, `.env.runpod`는 모델 실행 모드·키·경로를 담당합니다.
`.env.runpod`는 `--env-file`로 uv가 읽으므로 CUDA와 HF 설정도 Python 시작 전 적용됩니다.
같은 이름의 기존 셸 환경변수가 있으면 셸 값이 우선합니다.
NPP 라이브러리의 상대 경로 때문에 위 명령은 저장소 루트에서 실행하세요.

이번 변경에서는 GPU `pyproject.toml`에 `llm` 의존성을 추가했습니다.
잠금 파일은 이 작업 환경에서 갱신하지 않았으므로, 첫 실행 시 대상 서버에서 uv가 `gpu/uv.lock`을 갱신합니다.
갱신 후 재현 실행에는 `--locked`를 붙일 수 있습니다.

```bash
# 영상 1개
uv run --project gpu --env-file .env.runpod audyssey batch samples/sample.mp4

# 최종 믹스·영상 렌더링 생략
uv run --project gpu --env-file .env.runpod audyssey batch /workspace/inputs/*.mp4 --no-render
```

기존 진입점 `bash script/run_test.sh /workspace/inputs/*.mp4`도 같은 uv 명령을 호출합니다.
GPU 없이 stub으로 확인할 때는 `uv run --extra ui audyssey batch samples/sample.mp4`를 사용합니다.
이 경우 `.env.runpod`를 읽지 않고 모델 설정은 기본 stub을 사용합니다(기존 셸의 설정은 적용됩니다).

## 결과와 latency

```text
/workspace/audyssey-results/run01/
  batch.csv                 입력, job_id, 상태, 영상별 전체 시간, 오류
  timings.csv               호출별 단계 시간과 모델 로딩·처리 시간
  <job_id>/
    job.json                소스 설명, 평가, 경고, 결과 경로
    tracks/                 소스별 트랙 WAV
    out/stems/              내보낸 트랙 (residual 포함)
    out/mix.wav
    out/video.mp4
```

배치는 한 프로세스에서 순차 실행하며 모델 캐시를 재사용합니다. 첫 영상에는 모델 로딩 시간이 포함됩니다.
`batch.csv.duration_s`는 입력 검증·복사부터 분석과 선택적 렌더링까지이며 uv 설치·프로세스 시작 시간은 제외합니다.
`timings.csv.analysis_total`은 렌더링을 제외한 분석 시간입니다. 중첩된 단계 시간을 모두 더하지 마세요.
상세 열 설명은 [실행 시간 기록](performance.md)을 참고합니다.

실패한 영상은 기록한 뒤 다음 영상을 처리하며, 하나라도 실패하면 종료 코드 1입니다.
Judge 불합격 등 경고는 `job.json`에서 확인합니다. 기본 입력 길이 제한은 30초입니다.
재실행하면 새 작업을 만들고 CSV에 추가하며 자동 재개나 건너뛰기는 하지 않습니다.
같은 출력 폴더에는 한 배치만 실행하세요.
기본 믹스에는 residual이 포함되므로 분리 품질은 개별 `tracks/` WAV로 확인하세요.

이 변경은 코드·설정 준비 단계이며 RunPod 실제 GPU 실행은 아직 검증하지 않았습니다.
