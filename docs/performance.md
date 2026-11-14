# 실행 시간 기록

`audyssey batch`로 서버 없이 실행할 수도 있습니다. CLI에서는 `$AUDYSSEY_WORK_DIR/timings.csv`(기본 `work/jobs/timings.csv`)에 기록하고, 영상별 전체 시간은 같은 폴더의 `batch.csv`에 기록합니다. [RunPod CLI 실행](runpod-cli.md)을 참고하세요.

Audyssey는 파이프라인 단계별 실행 시간을 main 서버의 작업 디렉터리에 CSV로 기록합니다. Docker Compose 기본 설정에서는 다음 파일입니다.

```text
/app/work/jobs/timings.csv
```

호스트에서 확인하려면 프로젝트 디렉터리에서 실행합니다.

```bash
sudo docker compose exec main cat /app/work/jobs/timings.csv
```

## 기록되는 단계

CSV에는 호출마다 한 행이 추가됩니다. 같은 작업에서 플래닝이나 분리가 재시도되면 각 호출이 별도 행으로 남습니다.

| `stage` | 기록 시점 |
|---|---|
| `analysis_total` | 업로드 후 분석 시작부터 최종 상태 저장까지 |
| `preprocess` | 영상 정보 확인, 오디오 추출, 파형 계산 |
| `plan` | 플래너의 소스 계획 호출 1회 |
| `validate` | 계획 검증과 검증 후 시각 힌트 확인 1회 |
| `locate` | SAM3의 한 소스 시각 힌트 위치 확인 1회 |
| `mask_video` | 분리에 쓸 SAM3 마스크 영상 생성 1회 |
| `separate` | SAM Audio 분리 호출 1회 |
| `judge` | SAM Audio Judge 평가 호출 1회 |
| `residual` | 소스 트랙을 제외한 residual 계산 1회 |
| `render` | 편집 결과 렌더링 1회 |

`source_id`, `round`, `attempt`는 해당 호출에 적용되는 소스와 재시도 번호입니다. 적용되지 않는 값은 빈 칸입니다. `status`는 `done`, `warning`, `failed` 중 하나입니다. Judge가 소스를 통과시키지 못했거나 계획 검증에서 이슈가 있으면 `warning`으로 기록되며, 이때도 파이프라인은 정해진 재시도 정책에 따라 계속될 수 있습니다.

## 시간 열 읽기

- `duration_s`: main이 관측한 호출 전체의 벽시계 시간입니다. remote 도구를 쓸 때는 HTTP 왕복과 서버 처리 시간이 포함됩니다. 첫 요청이면 모델 로딩도 포함됩니다.
- `model_load_s`: 실제 도구 서버에서 모델을 가져오거나 초기화한 시간입니다. 프로세스 안에서 이미 로드된 캐시 모델을 재사용하면 보통 0에 가까운 값으로 기록됩니다.
- `processing_s`: 실제 도구 서버에서 오디오/영상 준비, 전처리, 추론, 출력 저장에 든 시간입니다.

`model_load_s`와 `processing_s`는 SAM Audio, Judge, SAM3의 실제 도구 구현에서만 채워집니다. 플래너의 `plan`·`validate`는 OpenAI 호환 API를 호출하므로 해당 열은 비어 있습니다. stub 구현도 모델을 로드하지 않으므로 비어 있을 수 있습니다.

분리 후보 재정렬을 켜 둔 경우, SAM Audio가 내부에서 사용하는 Judge·CLAP 등의 로딩 시간과 계산 시간은 별도의 `judge` 행이 아니라 `separate` 행에 포함됩니다. `judge` 행은 분리 후 별도로 호출되는 Judge 평가를 나타냅니다.

도구 서버가 main과 다른 머신에 있어도 `Server-Timing` 응답 헤더로 두 세부 시간을 전달해 같은 CSV 행에 기록합니다. 따라서 CSV는 main 서버의 작업 볼륨에 저장되고, 도구 서버의 시간이 별도 파일로 저장되지는 않습니다.

단계는 서로 중첩될 수 있습니다. 예를 들어 `analysis_total`에는 `preprocess`, `plan`, `separate` 등의 시간이 들어가며, `validate` 안에 `locate`가 들어갈 수 있습니다. 각 행의 시간을 모두 더해 전체 시간을 계산하지 말고, 전체 시간은 `analysis_total` 행을 보세요.

CSV는 새 작업이 실행될 때 기존 파일에 이어 씁니다. 작업별로 비교하려면 `job_id`로 행을 필터링하면 됩니다. CSV 기록 실패는 애플리케이션 로그에 남고 분석 작업 자체는 계속 진행합니다.
