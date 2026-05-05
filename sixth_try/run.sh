#!/usr/bin/env bash
# sixth_try: tf_fn + pre-warm + malloc_trim — plateau targeting harness
#
# 사용법:
#   chmod +x run.sh && ./run.sh
#
# 스트레스 옵션 (누수 가속):
#   ./run.sh -- --opt1 --opt2 --opt3
#
# tcmalloc 활성화 (google-perftools 설치 후):
#   sudo apt install -y google-perftools
#   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ./run.sh
#
# 환경변수 오버라이드:
#   MALLOC_ARENA_MAX=4 ./run.sh
#   SIXTH_TRY_WEB_PORT=9000 ./run.sh

set -euo pipefail

# ── TF C++ 메모리 해제 관련 환경변수 ─────────────────────────────────────────
# BFC allocator 비활성화 → system malloc(=tcmalloc) 으로 교체
# tcmalloc은 free() 호출 시 TCMALLOC_RELEASE_RATE 설정에 따라 OS에 즉시 반환
export TF_ALLOCATOR_USE_BFC="${TF_ALLOCATOR_USE_BFC:-0}"

# tcmalloc이 LD_PRELOAD로 로드된 경우 OS 반환 속도 (높을수록 빠름, 기본 1)
export TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}"

# glibc per-thread arena 수 제한 (검증됨 ✓ — 90% 누수 감소)
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# CPU only — GPU 디바이스 완전 차단
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# Eigen 스레드 수 고정 (스레드풀 메모리 안정화)
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"

cd "$(dirname "$0")" || { echo "Cannot cd to script dir"; exit 1; }

# ── Python 3.10 확인 ──────────────────────────────────────────────────────────
PY="${PYTHON:-python3.10}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[sixth_try] Python 3.10 이 필요합니다."
  echo "  Ubuntu 22.04: sudo apt update && sudo apt install -y python3.10 python3.10-venv"
  exit 1
fi
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null; then
  echo "[sixth_try] Python 3.10 이 필요합니다. (현재: $($PY --version))"
  exit 1
fi

# ── venv 생성 / 활성화 ────────────────────────────────────────────────────────
if ! "$PY" -m venv .venv; then
  echo "[sixth_try] venv 생성 실패: sudo apt install -y python3.10-venv"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel --quiet
python -m pip install -r requirements.txt --quiet

# ── TF 로드 확인 ──────────────────────────────────────────────────────────────
if ! python -c "import tensorflow as tf; print('[sixth_try] TF', tf.__version__, 'OK')" 2>&1; then
  echo "[sixth_try] TensorFlow 로드 실패 (AVX 미지원 CPU?)."
  exit 1
fi

# ── 실행 설정 출력 ────────────────────────────────────────────────────────────
echo "[sixth_try] ------- 환경 설정 -------"
echo "[sixth_try] MALLOC_ARENA_MAX      = $MALLOC_ARENA_MAX"
echo "[sixth_try] TF_ALLOCATOR_USE_BFC  = $TF_ALLOCATOR_USE_BFC"
echo "[sixth_try] TCMALLOC_RELEASE_RATE = $TCMALLOC_RELEASE_RATE"
echo "[sixth_try] CUDA_VISIBLE_DEVICES  = '${CUDA_VISIBLE_DEVICES}' (GPU 차단)"
echo "[sixth_try] LD_PRELOAD            = '${LD_PRELOAD:-}'"
echo "[sixth_try] --------------------------"

exec python app.py "$@"
