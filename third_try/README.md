# third_try — upstream과 같은 패턴으로 RSS 재현

가변 배치, ROI+필터, `0.2s` 폴링, **1000 iter마다 `K.clear_session()`만** 호출합니다.  
**`clear_session` 뒤에는 모델을 다시 로드하지 않습니다** (upstream과 동일).

## Ubuntu에서 실행

```bash
cd third_try
chmod +x run.sh
./run.sh
```

설정 변경은 `app.py` 상단 상수(`LOG_PREFIX`, `CLEAR_EVERY`, `POLL_SLEEP` 등)를 수정합니다.

`./run.sh`는 기본으로 **`MALLOC_ARENA_MAX=2`** 를 export합니다(glibc malloc arena 상한, `second_try`와 같은 목적). 다른 값을 쓰려면 실행 전에 환경 변수로 덮어쓰면 됩니다.

## 웹 트렌드 그래프 (`second_try`와 동일 패턴)

실행 후 브라우저에서 **http://127.0.0.1:8765/** 를 열면 RSS 추이를 볼 수 있습니다.  
포트는 환경 변수 `THIRD_TRY_WEB_PORT`(기본 `8765`), 바인드 주소는 `THIRD_TRY_WEB_HOST`(기본 `0.0.0.0`)로 바꿀 수 있습니다.

## 로그

- `rss_log.jsonl`, `rss_log.csv` — 1초 간격 RSS
