# fourth_try — clear_session 후 모델 재빌드 비교 실험

`third_try`와 동일한 웹 UI/로그를 유지하되, 차이점은 하나입니다.

- `CLEAR_EVERY`마다 `K.clear_session()` + `gc.collect()` 후 **모델을 즉시 재빌드/compile**

즉, `third_try`(clear만 하고 모델 재사용)와 `fourth_try`(clear 후 재빌드)의
메모리 추세를 비교할 수 있습니다.

## 실행

```bash
cd fourth_try
chmod +x run.sh
./run.sh
```

옵션은 `third_try`와 동일합니다.

- `--opt1` — Conv 채널 확대
- `--opt2` — 세그먼트당 `predict_on_batch` 1회 추가
- `--opt3` — `predict_on_batch` 1회 추가(누적)

예: `./run.sh -- --opt1 --opt2 --opt3`

`run.sh`는 기본으로 `MALLOC_ARENA_MAX=2`를 설정합니다.

## 웹 확인

- 기본: http://127.0.0.1:8765/
- 환경 변수: `FOURTH_TRY_WEB_HOST`, `FOURTH_TRY_WEB_PORT`

## 로그

- `rss_log.jsonl`, `rss_log.csv`
