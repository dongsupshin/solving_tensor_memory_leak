# third_try — upstream과 같은 패턴으로 RSS 재현

가변 배치, ROI+필터, `0.2s` 폴링, **1000 iter마다 `K.clear_session()`만** 호출합니다.  
**`clear_session` 뒤에는 모델을 다시 로드하지 않습니다** (upstream과 동일).

## Ubuntu에서 실행

`./run.sh`는 **`MALLOC_ARENA_MAX=2`** 를 export한 뒤 venv·Python을 실행합니다 (glibc arena 상한).

```bash
cd third_try
chmod +x run.sh
./run.sh
```

설정 변경은 `app.py` 상단 상수(`LOG_PREFIX`, `CLEAR_EVERY`, `POLL_SLEEP` 등)를 수정합니다. Arena 없이 돌리려면 `python app.py`를 직접 실행하세요.

## 로그

- `rss_log.jsonl`, `rss_log.csv` — 1초 간격 RSS
