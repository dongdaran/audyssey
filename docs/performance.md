# 실행 시간과 CSV 해석

Audyssey는 입력별 배치 시간과 파이프라인 호출별 시간을 CSV로 기록합니다.
CLI 실행 방법은 [RunPod 배치 가이드](runpod-cli.md)를 참고하세요.

## 기록 위치와 전체 시간

두 파일은 `AUDYSSEY_WORK_DIR` 아래에 저장됩니다. 기본 경로는 `work/jobs/`이며, RunPod 설정 예제는 `/workspace/audyssey-results/run01/`입니다.

| 파일·행 | 측정 범위 | 용도 |
|---|---|---|
| `batch.csv`의 `duration_s` | 입력 검증·복사 → 분석 → 선택적 렌더링까지 | CLI에서 영상 하나를 처리한 전체 시간 |
| `timings.csv`의 `analysis_total` | 분석 그래프 실행부터 최종 작업 상태 저장까지 | 렌더링을 제외한 분석 시간 |
| `timings.csv`의 나머지 행 | 단계 또는 도구 호출 1회 | 병목·재시도·로딩 비용 확인 |

`batch.csv`는 CLI 배치에서만 생성합니다. uv의 의존성 설치, Python 프로세스 시작, 배치 공통 도구 초기화 시간은 영상별 시간에 포함되지 않습니다.
분석 전에 입력 검증이 실패하면 `batch.csv`에는 실패 행이 남지만, 해당 입력의 `job_id`와 분석 시간 행은 없을 수 있습니다.

Docker Compose에서는 작업 볼륨 안의 `/app/work/jobs/timings.csv`를 확인합니다.

```bash
docker compose exec main cat /app/work/jobs/timings.csv
```

## CSV 열

`batch.csv`에는 `input`, `job_id`, `status`, `duration_s`, `error`가 들어갑니다.
`timings.csv`는 아래 열로 호출을 구분합니다.

| 열 | 의미 |
|---|---|
| `timestamp_utc` | 측정 시작 시각(UTC) |
| `job_id` | 작업 식별자. `batch.csv` 및 작업 폴더와 연결 |
| `stage` | 측정 단계 |
| `source_id` | 소스 식별자. 소스와 무관한 단계는 빈 칸 |
| `round` | 해당 단계의 반복 회차 |
| `attempt` | 해당 소스의 시도 번호. 재생성 등으로 누적될 수 있음 |
| `status` | `done`, `warning`, `failed` |
| `duration_s` | 호출 전체의 벽시계 시간(초) |
| `model_load_s` | 모델 초기화 또는 캐시된 모델을 가져오는 데 걸린 시간(초) |
| `processing_s` | 구현에서 측정한 전처리·추론·후처리 등의 시간(초) |
| `model_id` | 해당 측정에 지정된 모델 식별자. 없으면 빈 칸 |

`round`와 `attempt`는 적용되는 호출에만 채웁니다. `warning`은 계획 검증 이슈나 Judge·생성 채점 불합격을 뜻하며, 정해진 정책에 따라 재시도하거나 경고를 남기고 계속할 수 있습니다.
`failed`는 측정 구간에서 예외가 발생했다는 뜻입니다. 최종 소스 상태와 실패 이유는 작업 폴더의 `job.json`에서 확인합니다.

## 기록되는 단계

| `stage` | 측정 대상 |
|---|---|
| `analysis_total` | 전체 분석 |
| `preprocess` | 영상 확인, 오디오 추출·무음 판정, 파형 계산 |
| `plan` | 소스 계획 API 호출과 응답 처리 |
| `validate` | 계획 검증과 필요한 시각 힌트 위치 확인 |
| `locate` | 시각 힌트 하나에 대한 SAM3 위치 확인 |
| `mask_video` | 분리에 사용할 마스크 영상 생성 |
| `separate` | 소스별 SAM Audio 분리 호출 |
| `judge` | 분리 트랙의 Judge 평가 호출 |
| `generate` | 소스별 오디오 생성 호출 |
| `score_generation` | 생성 트랙의 CLAP 채점 호출 |
| `residual` | 원본에서 분리 트랙을 뺀 나머지 계산 |
| `render` | 믹스·트랙 저장과 최종 영상 렌더링 |

실제로 실행한 단계만 기록합니다. 시각 힌트가 꺼져 있으면 `locate`·`mask_video`는 실행하지 않고, 무음 입력은 residual을 계산하지 않습니다.
플래닝·분리·생성이 재시도되면 호출마다 별도 행을 남깁니다. 생성과 채점은 `source_id`, `round`, `attempt`로 대응시킬 수 있습니다.

## 로딩 시간과 처리 시간

첫 모델 호출에는 다운로드된 가중치 읽기와 초기화 비용이 포함될 수 있습니다.
같은 프로세스에서 캐시된 모델을 재사용하면 `model_load_s`는 보통 0에 가깝습니다. 첫 영상과 후속 영상은 구분해서 비교하세요.

세부 시간은 SAM Audio, Judge, SAM3, 생성 모델과 CLAP의 실제 구현에서 채웁니다.
플래너 API와 stub에는 세부 시간이 비어 있을 수 있습니다. 빈 칸은 0초를 의미하지 않습니다.

생성의 `processing_s`에는 영상 읽기, 생성, 리샘플링, 크로스페이드, WAV 저장이 포함됩니다.
생성 채점은 오디오 준비, CLAP 임베딩, 유사도·margin 계산을 포함합니다.
비교할 다른 소스가 없으면 모델 호출 없이 채점을 통과하므로, 해당 행의 CLAP 로딩 시간은 비어 있습니다.

분리 후보 재정렬에서 내부적으로 실행하는 Judge·CLAP 비용은 `separate`에 포함됩니다.
별도 `judge` 행은 분리 후 파이프라인이 호출한 평가입니다.

시간은 GPU 커널 전용 측정값이 아니라 벽시계 시간입니다.
잠금 대기나 호출 부대 비용 등이 있어 `duration_s`가 `model_load_s + processing_s`와 일치하지 않을 수 있습니다.

## 원격 도구 실행

`real`은 CLI 또는 main 프로세스에서 직접 측정합니다.
`remote`에서는 도구 서버가 `Server-Timing` 헤더로 모델 로딩·처리 시간을 반환하고, 호출 측이 같은 CSV 행에 기록합니다.
이때 `duration_s`에는 HTTP 왕복과 서버 처리도 포함됩니다. CSV 저장 위치는 호출 측의 작업 폴더입니다.

## 비교할 때 주의할 점

- **중첩된 행을 합산하지 마세요.** `analysis_total`에는 개별 단계가, `validate`에는 `locate`가 포함될 수 있습니다. 전체 배치 처리 시간은 `batch.csv`, 분석 시간은 `analysis_total`을 봅니다.
- **재시도와 모델 설정을 함께 비교하세요.** 같은 영상이라도 소스 수, 재시도 수, 후보 수가 다르면 시간이 달라집니다.
- **재실행은 새 작업입니다.** CSV는 기존 파일에 추가되므로 `job_id`로 구분합니다. 같은 작업의 모든 행이 최종 선택된 트랙만을 뜻하지는 않습니다.
- **시간 기록과 품질 판정은 다릅니다.** `done`은 해당 측정 구간을 완료했다는 의미이며, 오디오 품질은 평가 결과와 실제 트랙으로 확인합니다.

`timings.csv` 저장 실패는 로그에 남기고 분석을 계속합니다. `batch.csv`의 저장 실패에는 같은 예외 처리 정책이 적용되지 않습니다.
측정 구현은 [timing.py](../src/audyssey/timing.py), 호출 범위는 [graph.py](../src/audyssey/graph.py), 입력별 범위는 [batch.py](../src/audyssey/batch.py)에서 확인할 수 있습니다.
