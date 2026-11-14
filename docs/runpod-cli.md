# RunPod CLI 배치 실행

RunPod PyTorch Pod에서 여러 영상의 오디오를 처리하고, 트랙·영상·latency CSV를 저장하는 방법입니다.
GPU 모델은 CLI 프로세스에서 직접 실행하며, 계획과 계획 검증은 설정한 플래너 API를 호출합니다. 웹 서버나 포트 공개는 필요하지 않습니다.

## 1. 실행 환경 준비

다음 항목이 필요합니다.

- CUDA 12.8용 PyTorch를 실행할 수 있는 NVIDIA 드라이버와 GPU
- git, uv, ffmpeg/ffprobe
- 플래너 API 키와 필요한 모델의 Hugging Face 접근 권한
- 입력·가중치·결과를 보관할 볼륨

GPU 환경은 [gpu/pyproject.toml](../gpu/pyproject.toml)로 관리하며, [Python 버전 파일](../gpu/.python-version)은 3.12를 지정합니다. Pod에 기본 설치된 torch 대신 `gpu/.venv`를 사용합니다.

필요한 프로그램이 없다면 RunPod의 root 셸에서 설치합니다.

```bash
apt-get update
apt-get install -y ffmpeg git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

저장소를 받고, 이후 명령은 모두 저장소 루트에서 실행합니다.

```bash
cd /workspace
git clone https://github.com/capstone-audissey-fall-2026/audyssey.git
cd audyssey
cp .env.runpod.example .env.runpod
```

사용하는 브랜치에 `audyssey batch`와 `.env.runpod.example`이 포함되어 있어야 합니다.

## 2. 설정과 가중치 준비

[.env.runpod.example](../.env.runpod.example)을 복사한 `.env.runpod`에서 다음 값을 채웁니다.

| 용도 | 설정 |
|---|---|
| 플래너 | `AUDYSSEY_PLANNER_API_KEY`, 필요하면 `AUDYSSEY_PLANNER_BASE_URL`과 `AUDYSSEY_PLANNER_MODEL` 변경 |
| 모델 접근 | `HF_TOKEN`: 필요한 gated·비공개 모델을 읽을 수 있는 토큰 |
| 가중치 저장 | `HF_HOME`: 지속 볼륨의 캐시 경로 |
| 결과 저장 | `AUDYSSEY_WORK_DIR`: 실행별 결과 폴더 |
| GPU 선택 | `CUDA_VISIBLE_DEVICES` |

예제는 GPU 도구를 `real`, 시각 힌트를 끈 상태로 설정합니다. `.env.runpod`는 git에서 제외됩니다.
모델별 권한은 [실제 모델 실행 준비](running.md#실제-모델로-실행하기-팀원용)를 확인하세요.

최초 실행 전 가중치를 다운로드합니다. `HF_TOKEN`을 설정했다면 별도 로그인 없이 사용할 수 있습니다.

```bash
HF_HUB_OFFLINE=0 uv run --project gpu --env-file .env.runpod bash scripts/download-weights.sh
```

[다운로드 스크립트](../scripts/download-weights.sh)는 생성용 비공개 모델과 SAM3 가중치도 받습니다.
실제 생성 구현은 캐시된 가중치와 `HF_HUB_OFFLINE=1`을 요구하므로, 다운로드 후에는 예제의 오프라인 설정을 유지합니다.

`uv run`은 Python 환경과 의존성을 준비한 뒤 명령을 실행합니다. 별도 `uv sync`나 `--with`는 필요하지 않습니다.
현재 `gpu/uv.lock`에는 `audyssey[ui]`, manifest에는 `audyssey[ui,llm]`이 지정되어 있어 첫 실행 시 잠금 파일 갱신이 필요합니다.
대상 서버에서 갱신한 잠금 파일로 재현 실행할 때는 `--locked`를 추가하세요.

## 3. 배치 실행

먼저 영상 하나로 확인합니다.

```bash
uv run --project gpu --env-file .env.runpod audyssey batch --mode separate samples/sample.mp4
```

여러 영상은 같은 명령에 경로를 나열하거나 셸 glob으로 전달합니다.

```bash
uv run --project gpu --env-file .env.runpod audyssey batch --mode separate /workspace/inputs/*.mp4
```

기본값은 분석 후 믹스·영상·트랙까지 내보내는 것입니다.

```bash
# 분석 후 최종 믹스·영상·트랙까지 저장
uv run --project gpu --env-file .env.runpod audyssey batch --mode separate /workspace/inputs/*.mp4

# 분석만 실행: 개별 트랙과 시간 기록은 저장하고 최종 렌더링은 생략
uv run --project gpu --env-file .env.runpod audyssey batch --mode separate /workspace/inputs/*.mp4 --no-render
```

`--no-render`를 붙여도 분석 중 만들어진 `tracks/`, `job.json`, 시간 CSV는 저장됩니다. 최종 내보내기인 `out/mix.wav`, `out/video.mp4`, `out/stems/`는 만들지 않습니다.

- 영상은 순차 처리하며 같은 프로세스의 모델 캐시를 재사용합니다.
- 재실행하면 새 작업을 만들고 CSV에 추가합니다. 자동 재개·중복 건너뛰기는 없습니다.
- 같은 결과 폴더에는 배치 프로세스 하나만 실행하세요.

`--env-file`은 CUDA·HF 설정을 Python 시작 전에 적용합니다. 기존 셸에 같은 환경변수가 있으면 셸 값이 우선합니다.
예제의 `LD_LIBRARY_PATH`가 상대 경로이므로 저장소 루트에서 실행해야 합니다.

### 분리와 생성 선택

`--mode`는 필수이며 `separate`, `generate` 중 하나를 지정합니다. 자동 선택 모드는 없습니다.

```bash
# 기존 오디오를 소스별 트랙으로 분리
uv run --project gpu --env-file .env.runpod audyssey batch --mode separate /workspace/inputs/*.mp4

# 원본 오디오 유무와 관계없이 소스별 새 오디오 생성
uv run --project gpu --env-file .env.runpod audyssey batch --mode generate /workspace/inputs/*.mp4

# 생성 트랙과 latency만 저장하고 최종 믹스·영상 내보내기 생략
uv run --project gpu --env-file .env.runpod audyssey batch --mode generate /workspace/inputs/*.mp4 --no-render
```

| 모드 | 실행 내용 | 최종 믹스 |
|---|---|---|
| `separate` | 기존 오디오를 SAM Audio로 분리. 무음 입력은 실패로 기록 | 분리 트랙과 residual |
| `generate` | 영상과 소스 설명을 바탕으로 소스마다 새 오디오 생성. 대사는 건너뜀 | 생성 트랙만 포함. 원본 오디오·residual은 포함하지 않음 |

생성 모드는 입력 영상의 오디오를 지우거나 입력을 수정하지 않습니다. 생성한 믹스를 다시 분리하는 방식도 아닙니다.
오디오 스트림이 없거나 최대 진폭이 −60dBFS 미만이면 무음으로 판정합니다.
두 결과를 비교하려면 `AUDYSSEY_WORK_DIR`가 다른 설정 파일을 사용하거나, 각 실행에서 출력된 `job_id`로 구분하세요.
이 모드 옵션은 배치 CLI에 적용하며, 웹 API의 기존 입력별 자동 경로 선택은 유지합니다.

## 4. 결과 확인

예제 설정에서는 다음 위치에 저장됩니다.

```text
/workspace/audyssey-results/run01/
  batch.csv                 입력별 job_id, 상태, 전체 시간, 오류
  timings.csv               단계별 시간, 모델 로딩·처리 시간
  <job_id>/
    job.json                소스 설명, 상태, 평가 결과, 산출물 경로
    tracks/                 소스별 트랙 WAV
    out/stems/              내보낸 트랙
    out/mix.wav             최종 믹스
    out/video.mp4           최종 영상
```

`--no-render`를 쓰면 `out/`의 최종 산출물은 만들지 않습니다.
분리 결과의 기본 믹스에는 residual이 포함되므로 원본과 비슷할 수 있습니다. 분리 품질은 `tracks/`의 개별 WAV로 확인하세요.

입력별 결과는 `batch.csv`에서 찾고, 같은 `job_id`로 `timings.csv`와 작업 폴더를 연결합니다.
latency의 측정 범위와 CSV 해석은 [실행 시간 기록](performance.md)을 참고하세요.

## 자주 막히는 곳

| 증상 | 확인할 것 |
|---|---|
| `ffmpeg` 또는 `ffprobe`를 찾지 못함 | Python 패키지와 별도로 ffmpeg 설치 필요 |
| `planner HTTP 400` | 오류 본문과 플래너 주소·모델·키 확인. 상태 코드만으로 원인을 단정할 수 없음 |
| `video is …; maximum is …` | 실제 길이가 제한 초과. 필요하면 `.env.runpod`의 `AUDYSSEY_MAX_DURATION_S` 조정(기본 30초) |
| 가중치를 찾지 못함 | `HF_HOME`, 모델 접근 권한, 사전 다운로드 완료 여부 확인 |
| `separate mode requires non-silent input audio` | 입력 오디오 확인. 무음 영상의 소리를 만들려면 `--mode generate` 사용 |
| 생성이 오프라인 설정을 요구함 | 다운로드를 완료하고 `HF_HUB_OFFLINE=1`로 실행 |

입력 처리 중 예외가 발생하면 `batch.csv`에 기록하고 다음 입력으로 넘어가며, 하나라도 이런 실패가 있으면 종료 코드는 1입니다.
다만 소스별 실패는 그래프 내부에서 처리될 수 있으므로, 종료 코드나 배치 상태만으로 모든 트랙의 성공을 판단하지 말고 `job.json`의 소스·단계 상태도 확인하세요.

이 가이드는 CLI·설정·모델 로더 코드를 기준으로 작성했습니다. 문서 개정 과정에서 RunPod 설치나 실제 GPU 실행은 수행하지 않았습니다.
