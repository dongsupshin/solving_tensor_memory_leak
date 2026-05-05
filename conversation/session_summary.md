# thynC AI TF 메모리 누수 — 세션 요약 + 진행 중인 문제

> **목적**: Cursor 또는 다른 AI 에이전트에게 컨텍스트 + 진행 중 문제를 전달.
> **작성일**: 2026-05-05 (Claude와의 대화 정리)

---

## 1. 핵심 문제 한 문단

`C:\Users\dongs\workspace\seers\thync\ai\source\ai\upstream` 의 thynC AI 시스템(TensorFlow 2.10 CPU + Keras 2.10, Ubuntu 22.04, 24/7 의료기기 ECG 모니터링)의 SEGMENTATION 프로세스가 long-running 동안 RSS가 단조 증가하여 2GB 임계 도달 시 자동 재시작되는 문제. 현재 `MALLOC_ARENA_MAX=2` 적용으로 누수 속도 90% 감소(체감 10배)했으나 잔여 ~5.5 MB/call 누수가 남아 운영상 plateau 미달성. 의료기기라 SEG 재시작 동안 HR/RR/ST 등 임상 지표가 동시에 정지하는 문제도 동반. **목표는 "운영상 0 plateau" 달성 — 수 주~수 개월 무재시작 안정 운영.**

---

## 2. 시스템 환경

| 항목 | 값 |
|---|---|
| 도메인 | 의료기기 — 24/7 ECG 모니터링 |
| OS | Ubuntu 22.04 (Windows에선 동일 코드/데이터로 누수 미발생) |
| Python | 3.10 |
| TensorFlow | 2.10.0 CPU (`tensorflow-cpu==2.10.0`) |
| Keras | 2.10.0 |
| 모델 형식 | SavedModel (DDRNet 기반 1D segmentation) |
| 운영 패턴 | 0.2초 폴링, batch dim 1~50 가변, 4096 샘플(=16초) 입력 윈도우 |
| 샘플링 주파수 | 256 Hz |

---

## 3. 누수 4-Layer 분석 (Claude와 정립)

### Layer 1 — glibc per-thread arena fragmentation
- 기본 max arena = `min(8 × CPU코어, 128)`
- 각 arena가 freed 메모리를 캐싱, OS 미반환
- **`MALLOC_ARENA_MAX=2`로 잡힘. 누수 속도 90% 감소** ✓

### Layer 2 — tf.function trace 캐시 누적
- 가변 batch dim → 호출마다 새 ConcreteFunction trace
- 캐시 자동 정리 안 됨, K.clear_session()도 부분적
- **Javid 패턴(input_signature)으로 잡힐 가능성 높음**

### Layer 3 — TF BFC Allocator
- TF 자체 메모리 관리자, 한 번 받은 메모리 OS 미반환 (설계상)
- max input shape에 high water mark 고정
- **Fixed batch padding으로 plateau 보장 가능**

### Layer 4 — Eigen CPU 스레드 풀
- thread-local 메모리 풀, 스레드 lifetime 동안 유지
- Pre-warm으로 안정화 가능

---

## 4. 시도한 처방과 결과

| 처방 | 결과 |
|---|---|
| `MALLOC_ARENA_MAX=2` (또는 코드의 `mallopt(M_ARENA_MAX, 2)`) | **10배 완화 ✓** |
| `LD_PRELOAD=jemalloc` / `tcmalloc` | 부분 효과 |
| `TF_RUN_EAGER_OP_AS_FUNCTION=0` | 적용 중 |
| `predict()` → `predict_on_batch()` | 기본 적용 |
| `tf.convert_to_tensor()` 사전 변환 | 기본 적용 |
| `K.clear_session()` 1000-iter 주기 | **+267MB 부작용** 관측 (15/15회) |
| 2GB 임계 자동 재시작 안전망 | 운영 안정성 확보, 단 임상 지표 결측 발생 |

---

## 5. 핵심 관측 데이터 (실제 운영)

PID 1126739 측정:
- 시작 RSS 916MB → 종료 4276MB (10분에 +3360MB)
- 마지막 50회 predict_batch에서 +277MB → **호출당 ~5.5MB**

**`K.clear_session()` 부작용** (15/15회 모두 RSS 상승):
```
call=1000: 99 → 357 MB (+258)
call=2000: 109 → 387 MB (+278)
call=3000: 132 → 405 MB (+273)
... 평균 +267 MB
```

---

## 6. 미시도 카드 (실험 예정)

### A. A.H. Javid 패턴 (TF 커뮤니티 best practice)

```python
@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, 4096, 1], dtype=tf.float32)],
    reduce_retracing=True,
)
def _predict_fn(x_tensor):
    return self.model(x_tensor, training=False)

def predict_batch(self, signals):
    signals = signals.reshape(...)
    tf_arr = tf.convert_to_tensor(signals, dtype=tf.float32)
    return self._predict_fn(tf_arr).numpy()
```

핵심:
- `predict_on_batch` → `model(x, training=False)` 직접 호출
- `@tf.function(input_signature=...)`로 단일 ConcreteFunction 강제
- `predict_on_batch`의 Python-level 누수 우회

### B. Fixed Batch Padding (BFC plateau 보장)

```python
TARGET_BATCH = 64  # 또는 운영 max에 맞춤

def predict_batch(self, signals):
    actual_n = signals.shape[0]
    if actual_n < TARGET_BATCH:
        pad = np.zeros((TARGET_BATCH - actual_n, 4096), dtype=signals.dtype)
        signals_padded = np.concatenate([signals, pad], axis=0)
    elif actual_n > TARGET_BATCH:
        return self._predict_chunked(signals, TARGET_BATCH)
    else:
        signals_padded = signals
    
    tf_arr = tf.convert_to_tensor(
        signals_padded.reshape(TARGET_BATCH, 4096, 1),
        dtype=tf.float32
    )
    result = self._predict_fn(tf_arr)
    return result[:actual_n].numpy()  # 실제 환자분만 반환
```

기대 효과: BFC가 첫 호출에 max 메모리 잡고 plateau, 절대 안 자람.

### C. Pre-warm (Eigen 안정화)

```python
def warmup(self):
    dummy = np.zeros((TARGET_BATCH, 4096, 1), dtype=np.float32)
    tf_arr = tf.convert_to_tensor(dummy)
    for _ in range(20):
        self._predict_fn(tf_arr)
```

### D. 모델 재빌드 (잠재 진범 — third_try 실험 중)

second_try와 사용자 upstream의 결정적 차이 발견:
- second_try: `K.clear_session()` 후 **모델 재로드**
- upstream: clear_session만, 모델 재로드 X

→ 사용자 환경에서 1000 iter마다 `tf.keras.models.load_model(model_path)` 재실행 추가하면 trace 캐시까지 reset돼서 누수 잡힐 가능성.

---

## 7. 진행 중인 실험 — third_try

`C:\Users\dongs\workspace\solving_tensor_memory_leak\third_try\`

**구성**:
- `app.py` — standalone 시뮬레이션 (FastAPI 없음, ablation 모드 argparse)
- `run.sh` (Linux/WSL), `run.bat` (Windows)
- `requirements.txt` — TF 2.10.0 핀
- `README.md` — 실행 가이드

**Ablation 모드** (각 변수의 누수 기여도 격리):
```bash
./run.sh                                       # realistic (누수 기대)
./run.sh -- --rebuild-after-clear              # 모델 재빌드 추가
./run.sh -- --fixed-batch 16                   # 고정 batch
./run.sh -- --no-roi                           # ROI 슬라이싱 제거
./run.sh -- --no-clear-session                 # clear_session 제거
MALLOC_ARENA_MAX=2 ./run.sh                    # arena cap
```

**RSS 로깅**: 1초 간격으로 `rss_log.jsonl`, `rss_log.csv`

**기대 결과 매트릭스**:

| 실험 | 예상 RSS 곡선 | 의미 |
|---|---|---|
| 1. realistic | ~1-3 MB/분 단조 증가 | 누수 재현 |
| 2. + rebuild-after-clear | 상승 둔화 | 모델 재빌드 효과 |
| 3. + arena cap | 부분 둔화 | glibc 영향 |
| 4. 둘 다 (2+3) | plateau | second_try와 동등 |

---

## 8. second_try vs upstream 차이점 (코드 분석 결과)

| 항목 | second_try | 사용자 upstream | 위험도 |
|---|---|---|---|
| MALLOC_ARENA_MAX | **2** | 미설정 | 🔴 |
| 배치 크기 | 고정 16 | 가변 1~50 | 🔴 |
| 입력 shape | 2D 공간 패치 | 1D ECG (768,1) | 🟡 |
| 데이터 소스 | numpy 난수 | container_manager + filters | 🟡 |
| **K.clear_session 후 모델 재빌드** | **재빌드 O** | **재빌드 X** | 🔴🔴🔴 |
| try/except 구조 | 단일 | 환자별+배치별 | 🟡 |
| 폴링 sleep | ~0.011s | 0.2s | 🟢 |

**결정적 차이 — 모델 재빌드 유무**가 가장 의심.

**first_try ↔ second_try 차이**: app.py는 100% 동일. `run.sh` 3줄과 `Dockerfile` 1줄에 `MALLOC_ARENA_MAX=2`만 추가됨. 즉 first_try의 누수는 순전히 환경변수 미설정 때문.

---

## 9. 상위 코드 위치 참조

**핵심 추론 코드**:
- `C:\Users\dongs\workspace\seers\thync\ai\source\ai\upstream\ALGORITHM\SEGMENTATION\IPM_SEGMENTATION.py`
  - L70: `tf.keras.models.load_model(path)` (생성자 1회 로드)
  - L106: `predict_single` 안의 `model.predict()` — 누수 큰 패턴
  - L122-153: `predict_batch` (가변 batch, predict_on_batch, del cleanup 성공 경로에만)
  - L136-141: `iterCnt % 1000 == 0`에 `K.clear_session()` (모델 재빌드 없음)

**프로세스 루프**:
- `C:\Users\dongs\workspace\seers\thync\ai\source\ai\upstream\ALGORITHM\SEGMENTATION\IPM_SEGMENTATION_PROCESS.py`
  - L45: `container_manager.get_roi(id, (edIdx-4097, edIdx-1), ECGVOLT)` — 4096 샘플 ROI
  - L65-66: `signals = np.array(FEcgROIs); predict_batch(signals)`

**안전망**:
- `C:\Users\dongs\workspace\seers\thync\ai\source\ai\upstream\IPM_CONTROL_MODULE.py`
  - L522-622: `check_memorylimit()` — RSS 임계 체크 + 재시작 트리거
  - L362-386: `restart_process()` — terminate + 새 인스턴스 spawn

**메모리 프로빙**:
- `source/ai/log/ipm_mem_probe.jsonl` — 운영 환경 측정 데이터
- `algorithm_setting.ini` — `M_ARENA_MAX`, `CLEAR_SESSION_EVERY_N_PREDICT_BATCHES`, `TRACEMALLOC_SNAPSHOT_EVERY_N_SEGMENT`, `PREDICT_BATCH_LOG_EVERY_N` 제어

---

## 10. Cursor에게 부탁하는 작업

### 우선순위 1 — third_try ablation 실험 실행

`C:\Users\dongs\workspace\solving_tensor_memory_leak\third_try\` 에서:

```bash
# WSL 또는 Linux
cd third_try

# 실험 1: 누수 재현
./run.sh > exp1_realistic.log 2>&1
# rss_log.csv를 exp1_rss.csv로 백업

# 실험 2: 모델 재빌드 추가 (가장 결정적)
./run.sh -- --rebuild-after-clear > exp2_rebuild.log 2>&1
# rss_log.csv를 exp2_rss.csv로 백업

# 실험 3: arena cap만
MALLOC_ARENA_MAX=2 ./run.sh > exp3_arena.log 2>&1

# 실험 4: 둘 다 (second_try 동등)
MALLOC_ARENA_MAX=2 ./run.sh -- --rebuild-after-clear > exp4_both.log 2>&1
```

각 30분 이상 실행. 종료 후 4개 RSS 곡선 비교 — **2번이 plateau 가까우면 모델 재빌드가 진범 확정**.

### 우선순위 2 — Javid 패턴을 third_try에 추가

현재 `third_try/app.py`에 ablation 모드 더 추가:
- `--javid-pattern` — `model(x, training=False)` 직접 호출 + `@tf.function(input_signature=...)`
- `--fixed-padding TARGET` — 항상 TARGET batch로 padding

### 우선순위 3 — 우승 조합을 사용자 upstream에 적용

ablation 결과로 진범 확정되면 다음을 `IPM_SEGMENTATION.py`에 적용:

```python
# IPM_SEGMENTATION.py 의 predict_batch 변경

class IPM_SEGMENTATION:
    def __init__(self, path, ...):
        self.model_path = path
        with tf.device(self.deviceType):
            self.model = tf.keras.models.load_model(path)
        self.iterCnt = 0
        
        # Javid 패턴 ConcreteFunction
        @tf.function(
            input_signature=[tf.TensorSpec(shape=[None, 4096, 1], dtype=tf.float32)],
            reduce_retracing=True,
        )
        def _predict_fn(x_tensor):
            return self.model(x_tensor, training=False)
        self._predict_fn = _predict_fn
    
    def predict_batch(self, signals):
        try:
            signals = signals.reshape(signals.shape[0], signals.shape[1], 1)
            with tf.device(self.deviceType):
                tf_arr = tf.convert_to_tensor(signals, dtype=tf.float32)
                _predict = self._predict_fn(tf_arr).numpy()
        finally:
            try:
                del tf_arr
            except NameError:
                pass
        
        self.iterCnt += 1
        # ablation 결과에 따라 결정 — 일단 0으로 끔 (부작용 관측)
        if 0 < self.clear_session_every_n and self.iterCnt % self.clear_session_every_n == 0:
            K.clear_session()
            gc.collect()
            # 모델 재로드 추가 (third_try 실험 2 결과 반영)
            with tf.device(self.deviceType):
                self.model = tf.keras.models.load_model(self.model_path)
        
        return _predict
```

### 우선순위 4 — 의료기기 fail-over 패턴

SEG 재시작 동안 HR/RR 결측 방지:
- Hot Spare 패턴 (대기 인스턴스 1개 미리 spawn)
- 또는 UI Stale Detection (`>5초 갱신 없으면 "—" 표시`)

### 우선순위 5 — 진단 도구 셋업 (잔여 누수 정체 확정)

3-Track 동시 측정:
1. **tracemalloc** — Python heap 누적 추적
2. **`/proc/PID/smaps`** — `[anon]` vs `[heap]` 메모리 영역 분리
3. **gperftools heap profiler** — C++ symbol 단위 누수 추적
   ```bash
   sudo apt install google-perftools
   HEAPPROFILE=/tmp/seg_heap \
   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so \
   python upstream/IPM_MAIN.py
   google-pprof --text /tmp/seg_heap.0001.heap | head -30
   ```

진단 매트릭스:

| tracemalloc | smaps `[anon]` | glibc heap | 결론 |
|---|---|---|---|
| 평탄 | 자람 | 평탄 | TF BFC 누수 |
| 평탄 | 평탄 | 자람 | glibc fragmentation 잔존 |
| 자람 | — | — | Python 객체 누적 (predict_on_batch 등) |
| 평탄 | 평탄 | 평탄 | 누수 사라짐 ✓ |

---

## 11. 참고 자료

### Claude와의 대화에서 작성된 문서들 (참고용)
이전에 Claude가 만든 문서들 — 필요 시 사용자가 따로 보관 중:
- `thync_tf_memory_leak_investigation.md` — 누수 4-layer 분석 + 권장 처방
- `thync_glibc_memory_issue.md` — glibc fragmentation + mallopt 패치
- `thync_tf58676_consultation_brief.md` — TF #58676 이슈 컨설팅 브리핑
- `thync_memory_leak_current_state_brief.md` — 현재 상태 브리핑 (다른 AI용)
- `thync_tf_leak_experiment_runbook.md` — 실험 런북

### GitHub 이슈 (관련)
- [#58676 — Higher Memory Usage with model.predict in Recent TF Versions](https://github.com/tensorflow/tensorflow/issues/58676)
- [#44711 — memory leak in tf.keras.Model.predict](https://github.com/tensorflow/tensorflow/issues/44711)
- [#13118 (keras) — Repeatedly calling model.predict results in memory leak](https://github.com/keras-team/keras/issues/13118)
- [#33030 — Memory leak on TF 2.0 with model.predict](https://github.com/tensorflow/tensorflow/issues/33030)
- [#38598 — Memory-leak-like behavior on subsequent predictions (since TF 2.1.0)](https://github.com/tensorflow/tensorflow/issues/38598)

### 외부 사례
- A.H. Javid 블로그 (TF 2.19 + retracing 0회 사례, 메모리 -45%, latency -73%): https://ahjavid.github.io/technical-notes-blog/posts/tensorflow-retracing-optimization/
- The Kernel Trip (정직한 결론: "Beware, none of them actually works"): https://www.thekerneltrip.com/python/keras-memory-leak/

---

## 12. 결론 한 줄

> **TF 2.10 + ARENA=2 적용 후에도 잔여 ~5.5 MB/call이 남아있고, 이를 잡으려면 third_try ablation 실험으로 진범 확정 후 Javid 패턴 + Fixed Padding + (검증되면) 모델 주기적 재빌드 조합을 사용자 upstream에 적용. 의료기기 안전 측면은 Hot Spare 패턴으로 SEG 재시작 시 HR/RR 결측 방지 병행.**

---

## 13. 주요 대화 흐름 요약 (시간 순)

1. **초기 조사** — TF 2.10 누수가 보고되는 패턴인지 확인. predict() 사용을 의심
2. **코드 분석** — `predict_on_batch` 사용 중이라 일부 워크어라운드 적용된 상태였음 발견
3. **`K.clear_session()` 부작용 발견** — 호출 시 +267MB 스파이크 (15/15회)
4. **glibc fragmentation 가설 → ARENA=2 검증** → 10배 완화 입증
5. **잔여 ~5.5MB/call 남아있음 인식** — 4-layer 분석 정립
6. **A.H. Javid 패턴 발견** — input_signature + model() 직접 호출이 retrace 누수 차단의 best practice
7. **TF Serving 검토** — 누수 80-95% 감소 가능하나 over-engineering
8. **TF 2.8.2 다운그레이드 검토** — 단기 효과 크지만 EOL/SavedModel 호환성 부담
9. **2026-05-05 (이번 세션)**: third_try ablation 환경 셋업 완료. 모델 재빌드가 결정적 차이일 가능성 발견
