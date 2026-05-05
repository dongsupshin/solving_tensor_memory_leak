# fifth_try — #58676 원인 분리 진단 (Python vs TF/native)

`third_try`를 기반으로, 메모리 증가의 근본 원인이 Python 힙인지 TensorFlow/native(C++) 쪽인지 **판단 가능**하도록 만든 실험 버전입니다.

## 핵심 아이디어

- 동일한 workload에서 추론 엔진만 바꿔 비교
  - `--engine predict`: `model.predict_on_batch` 경로
  - `--engine tf_fn`: 단일 `@tf.function` traced `model(x)` 경로 (기본값, #58676 회피 후보)
- 웹/API에 메모리 분해 지표 노출
  - `rss_mb`, `uss_mb`
  - `py_tracemalloc_cur_mb`, `py_tracemalloc_peak_mb`
  - `native_est_mb = rss - py_tracemalloc_current` (근사치)
  - `gc_objects`, `allocated_blocks`
  - `growth_class` 자동 분류 (`likely_python_heap_growth` / `likely_native_or_tf_allocator_growth` / `mixed_or_inconclusive`)

## 실행

```bash
cd fifth_try
chmod +x run.sh
./run.sh
```

부하 옵션은 기존과 동일:

- `--opt1` — Conv 채널 확대
- `--opt2` — 세그먼트당 추론 1회 추가
- `--opt3` — 세그먼트당 추론 1회 추가(누적)

## 권장 비교 시나리오

같은 조건(`--opt3` 등)에서 엔진만 바꿔 실행:

```bash
./run.sh -- --opt3 --engine predict
./run.sh -- --opt3 --engine tf_fn
```

그 후 `/api/memory`의 `growth_class`, `diag.native_est_mb`, `diag.py_tracemalloc_cur_mb` 추세를 비교하세요.

## 웹 확인

- 기본: http://127.0.0.1:8765/
- 환경 변수: `FIFTH_TRY_WEB_HOST`, `FIFTH_TRY_WEB_PORT`

## 로그

- `rss_log.jsonl`, `rss_log.csv`
- CSV 컬럼: `ts_unix,elapsed_s,rss_mb,py_cur_mb,native_est_mb,iter`
