#!/usr/bin/env python3
"""
sixth_try/app.py — plateau-targeting harness.

Goal: TF C++ BFC allocator가 사용 후 메모리를 OS에 반환하도록 강제,
      + 단일 traced @tf.function(input_signature=...) + pre-warm 조합으로 RSS plateau 달성.

핵심 메커니즘:
  1. MALLOC_ARENA_MAX=2        — glibc arena fragmentation 차단 (검증됨 ✓)
  2. TF_ALLOCATOR_USE_BFC=0   — TF CPU 기본 allocator를 BFC→ system malloc으로 교체,
                                 tcmalloc이 있으면 tcmalloc이 free 즉시 OS 반환
  3. TCMALLOC_RELEASE_RATE=10 — tcmalloc이 LD_PRELOAD되면 1초 내 OS 반환 강제
  4. @tf.function(input_signature=[None-batch, L, 1]) — 단일 ConcreteFunction, retrace 0
  5. Pre-warm 20회 — Eigen 스레드풀 + BFC high-water mark 초기 안정화
  6. K.clear_session() 제거   — +267 MB 스파이크 유발, 제거가 최선
  7. gc.collect() 주기 호출  — Python 레퍼런스 사이클 청소

메모리 분해 지표 실시간 로깅:
  rss_mb, uss_mb, py_tracemalloc_cur_mb, native_est_mb, gc_objects

웹 UI: http://127.0.0.1:8766/  (다중 라인 차트: RSS / Python heap / Native 추정)
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import os
import sys
import threading
import time
import tracemalloc
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# ── TF 환경변수 — import 전에 설정해야 적용됨 ────────────────────────────────
# BFC allocator를 끄고 system malloc(=tcmalloc이 LD_PRELOAD되면 tcmalloc)으로 교체.
# tcmalloc은 free() 시 TCMALLOC_RELEASE_RATE에 따라 OS에 즉시 반환.
os.environ.setdefault("TF_ALLOCATOR_USE_BFC", "0")
# CPU only — GPU 디바이스 완전 차단
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Eigen thread 안정화
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
# tcmalloc release rate (LD_PRELOAD로 로드된 경우 유효)
os.environ.setdefault("TCMALLOC_RELEASE_RATE", "10")
# glibc arena cap
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# ── Constants (upstream SEGMENTATION_PARAM 기준) ──────────────────────────────
ECG_INPUT_NUM   = 768       # 256 * 3 샘플
ECG_SAMPLE_RATE = 256
ECG_BUFFER_LEN  = 20_000
N_PATIENTS      = 50
N_CLASSES       = 6

LOG_PREFIX   = "rss_log"
BASE_CH      = 32
POLL_SLEEP   = 0.2
CLEAR_EVERY  = 0           # 0 = clear_session 비활성화 (기본값; +267MB 스파이크 방지)
MAX_PATIENTS = N_PATIENTS
WARMUP_STEPS = 20          # pre-warm 호출 횟수

GC_EVERY     = 100         # N iter마다 gc.collect() — Python 레퍼런스 사이클 청소

WEB_HOST = os.environ.get("SIXTH_TRY_WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("SIXTH_TRY_WEB_PORT", "8766"))


@dataclass(frozen=True)
class RunConfig:
    log_prefix:   str   = LOG_PREFIX
    base_ch:      int   = BASE_CH
    poll_sleep:   float = POLL_SLEEP
    clear_every:  int   = CLEAR_EVERY
    max_patients: int   = MAX_PATIENTS
    warmup_steps: int   = WARMUP_STEPS
    gc_every:     int   = GC_EVERY
    opt1:         bool  = False   # wider model (+channels)
    opt2:         bool  = False   # +1 extra infer call/segment
    opt3:         bool  = False   # +1 more infer call/segment


# ── Global state ──────────────────────────────────────────────────────────────
_iter_count    = 0
_state_lock    = threading.Lock()
_stop_event    = threading.Event()
_rss_history:  Deque[tuple[float, float, float, float]] = deque()
# tuple: (elapsed_s, rss_bytes, py_cur_bytes, native_est_bytes)
_worker_error: str | None = None
_latest_diag:  dict       = {}


# ── tcmalloc 존재 확인 + malloc_stats 출력 ────────────────────────────────────
def _try_load_tcmalloc() -> bool:
    """LD_PRELOAD 없이 tcmalloc을 직접 로드 시도. 성공하면 True."""
    for lib in ["libtcmalloc_minimal.so.4", "libtcmalloc.so.4",
                "libtcmalloc_minimal.so", "libtcmalloc.so"]:
        path = ctypes.util.find_library(lib.replace(".so.4", "").replace(".so", ""))
        # find_library가 못 찾아도 직접 시도
        for candidate in ([path] if path else []) + [lib]:
            if not candidate:
                continue
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                print(f"[sixth_try] tcmalloc loaded: {candidate}", flush=True)
                return True
            except OSError:
                continue
    return False


def _malloc_trim() -> None:
    """glibc malloc_trim() 호출 — free된 힙 메모리를 OS에 반환."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(ctypes.c_size_t(0))
    except Exception:
        pass


# ── 메모리 진단 스냅샷 ─────────────────────────────────────────────────────────
def _diag_snapshot(proc: psutil.Process) -> dict:
    rss = float(proc.memory_info().rss)
    uss = 0.0
    try:
        uss = float(proc.memory_full_info().uss)
    except Exception:
        pass

    py_cur = py_peak = 0.0
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        py_cur, py_peak = float(cur), float(peak)

    native_est = max(0.0, rss - py_cur)

    return {
        "rss_mb":                 rss        / 1024 / 1024,
        "uss_mb":                 uss        / 1024 / 1024,
        "py_tracemalloc_cur_mb":  py_cur     / 1024 / 1024,
        "py_tracemalloc_peak_mb": py_peak    / 1024 / 1024,
        "native_est_mb":          native_est / 1024 / 1024,
        "gc_objects":             len(gc.get_objects()),
        "allocated_blocks":       sys.getallocatedblocks(),
    }


# ── 합성 ECG 필터 ──────────────────────────────────────────────────────────────
def _apply_filters(ecg_roi: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    if ecg_roi is None or len(ecg_roi) < 10:
        return None
    sig = ecg_roi.astype(np.float64)
    k   = max(3, sample_rate // 50)
    lp  = np.convolve(sig, np.ones(k, dtype=np.float64) / k, mode="same")
    lp -= np.mean(lp)
    return lp


# ── Fake patient / container ──────────────────────────────────────────────────
class _FakePatient:
    def __init__(self, pid: int, rng: np.random.Generator) -> None:
        self.pid = pid
        t      = np.linspace(0, ECG_BUFFER_LEN / ECG_SAMPLE_RATE, ECG_BUFFER_LEN)
        hr_hz  = rng.uniform(50, 100) / 60.0
        buf    = (
            np.sin(2 * np.pi * hr_hz * t) * 1.0
            + np.sin(2 * np.pi * hr_hz * 5 * t) * 0.3
            + np.sin(2 * np.pi * hr_hz * 3 * t) * 0.15
            + rng.standard_normal(ECG_BUFFER_LEN) * 0.05
        )
        self.ecg_buffer: np.ndarray = buf.astype(np.float64)
        self._end_idx: int = ECG_BUFFER_LEN

    def advance(self, n: int = 50) -> None:
        self._end_idx += n

    def get_ecg_index(self):
        return [(None, self._end_idx, None, None, None)]

    def get_roi(self, start: int, end: int) -> np.ndarray:
        buf, buf_len = self.ecg_buffer, len(self.ecg_buffer)
        length = end - start
        if length <= 0:
            return np.zeros(ECG_INPUT_NUM, dtype=np.float64)
        sc, ec = start % buf_len, end % buf_len
        if ec > sc:
            roi = buf[sc:ec]
        elif ec == sc:
            roi = buf[sc: sc + length]
        else:
            roi = np.concatenate([buf[sc:], buf[:ec]])
        if len(roi) < length:
            roi = np.pad(roi, (0, length - len(roi)))
        return roi[:length].copy()


class FakeContainerManager:
    def __init__(self, n_patients: int, rng: np.random.Generator) -> None:
        self._patients: dict = {i: _FakePatient(i, rng) for i in range(n_patients)}
        self._results:  dict = {}
        self._lock = threading.Lock()

    def get(self, patient_id: int, param: str):
        with self._lock:
            p = self._patients.get(patient_id)
        if p is None:
            return None
        if param == "ECGINDEX":
            return p.get_ecg_index()
        return None

    def get_roi(self, patient_id: int, index_range: tuple, param: str) -> Optional[np.ndarray]:
        with self._lock:
            p = self._patients.get(patient_id)
        if p is None:
            return None
        return p.get_roi(*index_range)

    def set(self, seg_data: dict, param: str) -> None:
        with self._lock:
            self._results[seg_data.get("patient_id", 0)] = seg_data

    def active_ids(self, rng: np.random.Generator, max_n: int) -> list:
        with self._lock:
            all_ids = list(self._patients.keys())
        n      = int(rng.integers(1, min(max_n, len(all_ids)) + 1))
        chosen = rng.choice(len(all_ids), size=n, replace=False)
        return [all_ids[i] for i in chosen]

    def stream_advance(self, n: int = 50) -> None:
        with self._lock:
            patients = list(self._patients.values())
        for p in patients:
            p.advance(n)


# ── Model builder ─────────────────────────────────────────────────────────────
def _build_ecg_segmentation_model(ecg_length: int, base_ch: int):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(ecg_length, 1), name="ecg_signal")
    x   = tf.keras.layers.Conv1D(base_ch,     7, padding="same", activation="relu")(inp)
    x   = tf.keras.layers.Conv1D(base_ch * 2, 5, padding="same", activation="relu")(x)
    x   = tf.keras.layers.Conv1D(base_ch * 2, 5, padding="same", activation="relu")(x)
    x   = tf.keras.layers.Conv1D(base_ch,     3, padding="same", activation="relu")(x)
    out = tf.keras.layers.Conv1D(N_CLASSES,   1, padding="same", activation="linear", name="logits")(x)
    return tf.keras.Model(inp, out, name="ecg_seg_dummy")


# ── RSS logger thread ─────────────────────────────────────────────────────────
def _rss_logger(log_prefix: str) -> None:
    proc = psutil.Process(os.getpid())
    t0   = time.time()
    with (
        open(log_prefix + ".jsonl", "w", encoding="utf-8") as fj,
        open(log_prefix + ".csv",   "w", encoding="utf-8") as fc,
    ):
        fc.write("ts_unix,elapsed_s,rss_mb,py_cur_mb,native_est_mb,iter\n")
        while not _stop_event.is_set():
            ts   = time.time()
            diag = _diag_snapshot(proc)
            rss_mb        = float(diag["rss_mb"])
            py_cur_mb     = float(diag["py_tracemalloc_cur_mb"])
            native_est_mb = float(diag["native_est_mb"])
            with _state_lock:
                it = _iter_count
                _latest_diag.update(diag)
            elapsed = ts - t0
            row = {
                "ts_unix":       round(ts, 3),
                "elapsed_s":     round(elapsed, 1),
                "rss_mb":        round(rss_mb, 2),
                "py_cur_mb":     round(py_cur_mb, 2),
                "native_est_mb": round(native_est_mb, 2),
                "iter":          it,
            }
            fj.write(json.dumps(row) + "\n")
            fc.write(f"{ts:.3f},{elapsed:.1f},{rss_mb:.2f},{py_cur_mb:.2f},{native_est_mb:.2f},{it}\n")
            fj.flush(); fc.flush()
            time.sleep(1.0)


# ── segment() ────────────────────────────────────────────────────────────────
def _segment(infer_fn, container_manager: FakeContainerManager,
             needed_ids: list, cfg: RunConfig) -> None:
    import tensorflow as tf

    ids, edIdxs, FEcgROIs = [], [], []

    for pid in needed_ids:
        try:
            indexes = container_manager.get(pid, "ECGINDEX")
            if indexes is None:
                continue
            _, edIdx, *_ = indexes[0]
            EcgROI  = container_manager.get_roi(pid, (edIdx - ECG_INPUT_NUM - 1, edIdx - 1), "ECGVOLT")
            FEcgROI = _apply_filters(EcgROI, ECG_SAMPLE_RATE)
            if FEcgROI is not None:
                ids.append(pid)
                edIdxs.append(edIdx)
                FEcgROIs.append(FEcgROI)
        except Exception:
            continue

    if not FEcgROIs:
        return

    try:
        signals = np.array(FEcgROIs, dtype=np.float64).reshape(-1, ECG_INPUT_NUM, 1)
        with tf.device("/CPU:0"):
            tf_arr   = tf.convert_to_tensor(signals, dtype=tf.float64)
            _predict = infer_fn(tf_arr)
            extra    = int(cfg.opt2) + int(cfg.opt3)
            for _ in range(extra):
                _predict = infer_fn(tf_arr)

        predicted_segments = np.array(_predict, copy=True)

        # 명시적 해제 — Python 레퍼런스 카운트 즉시 0으로
        del tf_arr, signals, _predict

    except Exception:
        return

    for pid, segment, edIdx in zip(ids, predicted_segments, edIdxs):
        try:
            pqrst    = np.argmax(segment, axis=1)
            seg_data = {"patient_id": pid, "pqrst": pqrst, "edIdx": edIdx}
            container_manager.set(seg_data, "SEGMENTATION")
        except Exception:
            continue

    del predicted_segments, ids, edIdxs, FEcgROIs, needed_ids


# ── Inference loop ────────────────────────────────────────────────────────────
def _inference_loop(cfg: RunConfig) -> None:
    global _iter_count, _worker_error

    import tensorflow as tf

    try:
        # ── TF 스레드 설정 (환경변수 이후 재적용) ────────────────────────────
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        # CPU only: GPU 디바이스 완전 차단 (GPU가 있어도 사용 안 함)
        tf.config.set_visible_devices([], "GPU")

        eff_ch = cfg.base_ch + (32 if cfg.opt1 else 0)
        with tf.device("/CPU:0"):
            model = _build_ecg_segmentation_model(ECG_INPUT_NUM, eff_ch)

        # ── 단일 ConcreteFunction — retrace 0 보장 ────────────────────────────
        # input_signature의 batch dim = None → 가변 batch(1명~수만명) 모두 단일 trace
        @tf.function(
            input_signature=[tf.TensorSpec(shape=[None, ECG_INPUT_NUM, 1], dtype=tf.float64)],
            reduce_retracing=True,
        )
        def infer_fn(x: tf.Tensor) -> tf.Tensor:
            with tf.device("/CPU:0"):
                return model(x, training=False)

        # ── Pre-warm: Eigen 스레드풀 + BFC high-water mark 초기 안정화 ────────
        print(f"[sixth_try] Pre-warming {cfg.warmup_steps} steps …", flush=True)
        with tf.device("/CPU:0"):
            dummy = tf.zeros([1, ECG_INPUT_NUM, 1], dtype=tf.float64)
            for _ in range(cfg.warmup_steps):
                _ = infer_fn(dummy)
        del dummy
        gc.collect()
        _malloc_trim()
        print("[sixth_try] Pre-warm done. Entering inference loop.", flush=True)

        rng               = np.random.default_rng(42)
        container_manager = FakeContainerManager(N_PATIENTS, rng)
        proc              = psutil.Process(os.getpid())

        # ── 설정 출력 ─────────────────────────────────────────────────────────
        print(f"[sixth_try] engine         = tf_fn  (single traced @tf.function, retrace=0)")
        print(f"[sixth_try] input_signature= [None, {ECG_INPUT_NUM}, 1] float64  → 가변 batch 단일 trace")
        print(f"[sixth_try] pre_warm       = {cfg.warmup_steps} steps")
        print(f"[sixth_try] clear_session  = {'every ' + str(cfg.clear_every) if cfg.clear_every > 0 else 'DISABLED (prevents +267MB spike)'}")
        print(f"[sixth_try] gc_every       = {cfg.gc_every} iters")
        print(f"[sixth_try] malloc_trim    = every {cfg.gc_every} iters  (glibc free→OS)")
        print(f"[sixth_try] MALLOC_ARENA_MAX = {os.environ.get('MALLOC_ARENA_MAX', '(unset)')}")
        print(f"[sixth_try] TF_ALLOCATOR_USE_BFC = {os.environ.get('TF_ALLOCATOR_USE_BFC', '(unset)')}")
        print(f"[sixth_try] TCMALLOC_RELEASE_RATE = {os.environ.get('TCMALLOC_RELEASE_RATE', '(unset)')}")
        print(f"[sixth_try] variable_batch = 1..{cfg.max_patients}  (fully supported)")
        print(f"[sixth_try] web            = http://127.0.0.1:{WEB_PORT}/")
        print(f"[sixth_try] log            = {cfg.log_prefix}.jsonl / .csv")
        sys.stdout.flush()

        loop_t0 = time.perf_counter()
        with _state_lock:
            _rss_history.clear()
            rss0 = float(proc.memory_info().rss)
            _rss_history.append((0.0, rss0, 0.0, rss0))

        iter_cnt    = 0
        t_start     = time.time()
        t_last_print = t_start

        while not _stop_event.is_set():
            container_manager.stream_advance(50)
            needed_ids = container_manager.active_ids(rng, max_n=cfg.max_patients)

            _segment(
                infer_fn=infer_fn,
                container_manager=container_manager,
                needed_ids=needed_ids,
                cfg=cfg,
            )

            iter_cnt += 1

            # ── 주기적 메모리 반환 ─────────────────────────────────────────────
            if iter_cnt % cfg.gc_every == 0:
                gc.collect()
                _malloc_trim()   # glibc: free된 힙 페이지 OS 반환
                # TF 내부 메모리 통계 리셋 (2.10에서 가용하면 실행)
                try:
                    tf.config.experimental.reset_memory_stats("CPU:0")
                except Exception:
                    pass

            rss = float(proc.memory_info().rss)
            diag = _diag_snapshot(proc)
            with _state_lock:
                _iter_count = iter_cnt
                _rss_history.append((
                    time.perf_counter() - loop_t0,
                    rss,
                    diag["py_tracemalloc_cur_mb"] * 1024 * 1024,
                    diag["native_est_mb"] * 1024 * 1024,
                ))
                _latest_diag.update(diag)

            now = time.time()
            if now - t_last_print >= 10.0:
                rss_mb = rss / 1024 / 1024
                print(
                    f"[{now - t_start:6.0f}s] iter={iter_cnt:6d}  "
                    f"rss={rss_mb:.1f}MB  "
                    f"py={diag['py_tracemalloc_cur_mb']:.1f}MB  "
                    f"native≈{diag['native_est_mb']:.1f}MB  "
                    f"batch={len(needed_ids)}",
                    flush=True,
                )
                t_last_print = now

            time.sleep(cfg.poll_sleep)

    except Exception as e:
        with _state_lock:
            _worker_error = repr(e)
        raise


# ── Web UI HTML ───────────────────────────────────────────────────────────────
def _html_page() -> str:
    return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>sixth_try — RSS plateau monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body   { font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3; }
    header { padding: 0.9rem 1.4rem; border-bottom: 1px solid #21262d;
             display: flex; align-items: center; gap: 1.2rem; flex-wrap: wrap; }
    h1     { font-size: 1rem; font-weight: 600; white-space: nowrap; }
    .badge { font-size: 0.75rem; padding: 0.2rem 0.55rem; border-radius: 999px;
             background: #21262d; color: #8b949e; white-space: nowrap; }
    .badge.ok  { background: #1a3a2a; color: #3fb950; }
    .badge.warn{ background: #3a2a0f; color: #d29922; }
    .stats { font-size: 0.82rem; color: #8b949e; margin-left: auto; }
    main   { padding: 1.1rem 1.4rem; display: grid;
             grid-template-columns: 1fr; gap: 1rem; max-width: 1200px; margin: 0 auto; }
    .card  { background: #161b22; border: 1px solid #21262d; border-radius: 8px;
             padding: 0.9rem 1rem; }
    .card h2 { font-size: 0.82rem; color: #8b949e; margin-bottom: 0.6rem; text-transform: uppercase; letter-spacing: .04em; }
    .chart-wrap { height: 340px; }
    .chart-wrap-sm { height: 220px; }
    .diag  { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 0.5rem; }
    .diag-item { background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
                 padding: 0.5rem 0.75rem; }
    .diag-item .label { font-size: 0.72rem; color: #8b949e; margin-bottom: 0.25rem; }
    .diag-item .value { font-size: 1.05rem; font-weight: 600; color: #58a6ff; }
    .error { color: #f85149; margin-top: 0.8rem; font-size: 0.85rem; white-space: pre-wrap; }
    @media (min-width: 900px) {
      main { grid-template-columns: 2fr 1fr; }
      .card.full { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
<header>
  <h1>sixth_try — RSS plateau monitor</h1>
  <span class="badge" id="badge-engine">engine: tf_fn</span>
  <span class="badge" id="badge-arena">ARENA_MAX=?</span>
  <span class="badge" id="badge-bfc">BFC=?</span>
  <span class="badge" id="badge-tcmalloc">tcmalloc=?</span>
  <span class="stats" id="stats">iter=— &nbsp; rss=—</span>
</header>
<main>
  <!-- 메인 차트: RSS / Python heap / Native 추정 -->
  <div class="card full">
    <h2>메모리 추이 (RSS · Python heap · Native/TF 추정)</h2>
    <div class="chart-wrap"><canvas id="chart-main"></canvas></div>
  </div>

  <!-- 증가율 차트 -->
  <div class="card">
    <h2>증가율 (MB / 분, 60s 이동평균)</h2>
    <div class="chart-wrap-sm"><canvas id="chart-rate"></canvas></div>
  </div>

  <!-- 진단 수치 -->
  <div class="card">
    <h2>실시간 진단</h2>
    <div class="diag">
      <div class="diag-item"><div class="label">RSS</div><div class="value" id="d-rss">—</div></div>
      <div class="diag-item"><div class="label">USS</div><div class="value" id="d-uss">—</div></div>
      <div class="diag-item"><div class="label">Python heap</div><div class="value" id="d-py">—</div></div>
      <div class="diag-item"><div class="label">Native/TF ≈</div><div class="value" id="d-native">—</div></div>
      <div class="diag-item"><div class="label">GC objects</div><div class="value" id="d-gc">—</div></div>
      <div class="diag-item"><div class="label">Iterations</div><div class="value" id="d-iter">—</div></div>
    </div>
    <p class="error" id="err"></p>
  </div>
</main>
<script>
// ── 차트 공통 옵션 ──────────────────────────────────────────────────────────
const gridColor = 'rgba(48,54,61,0.8)';
const mkChart = (id, datasets, yLabel) => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets },
  options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: { ticks: { color: '#8b949e', maxTicksLimit: 12 },
           grid: { color: gridColor },
           title: { display: true, text: 'Elapsed (s)', color: '#8b949e' } },
      y: { ticks: { color: '#8b949e' },
           grid: { color: gridColor },
           title: { display: true, text: yLabel, color: '#8b949e' } }
    },
    plugins: {
      legend: { labels: { color: '#e6edf3', boxWidth: 12 } },
      tooltip: { backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
                 titleColor: '#8b949e', bodyColor: '#e6edf3' }
    }
  }
});

const mainChart = mkChart('chart-main', [
  { label: 'RSS (MB)',            borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.07)',
    fill: true, tension: 0.2, pointRadius: 0, data: [] },
  { label: 'Python heap (MB)',    borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.05)',
    fill: false, tension: 0.2, pointRadius: 0, data: [] },
  { label: 'Native/TF est. (MB)', borderColor: '#d29922', backgroundColor: 'rgba(210,153,34,0.05)',
    fill: false, tension: 0.2, pointRadius: 0, data: [] },
], 'MB');

const rateChart = mkChart('chart-rate', [
  { label: 'RSS 증가율 (MB/min)',     borderColor: '#58a6ff', tension: 0.3, pointRadius: 0, data: [] },
  { label: 'Native 증가율 (MB/min)', borderColor: '#d29922', tension: 0.3, pointRadius: 0, data: [] },
], 'MB / min');

// ── 이동평균 증가율 계산 ────────────────────────────────────────────────────
const WINDOW_S = 60;
let prevSeries = [];

function computeRates(series) {
  if (series.length < 2) return { labels: [], rss: [], native: [] };
  const labels = [], rss = [], native = [];
  for (let i = 1; i < series.length; i++) {
    const [t1, r1, , n1] = series[i];
    // find point ~WINDOW_S seconds ago
    let j = i - 1;
    while (j > 0 && t1 - series[j][0] < WINDOW_S) j--;
    const [t0, r0, , n0] = series[j];
    const dt = t1 - t0;
    if (dt < 1) continue;
    labels.push(Number(t1).toFixed(1));
    rss.push(((r1 - r0) / dt / 1024 / 1024 * 60).toFixed(3));
    native.push(((n1 - n0) / dt / 1024 / 1024 * 60).toFixed(3));
  }
  return { labels, rss, native };
}

// ── 폴링 ───────────────────────────────────────────────────────────────────
async function tick() {
  try {
    const r = await fetch('/api/memory');
    const j = await r.json();

    // 상태 배지
    const arena = j.config?.malloc_arena_max ?? '?';
    const bfc   = j.config?.tf_allocator_use_bfc ?? '?';
    const tcm   = j.config?.tcmalloc_loaded ?? false;
    document.getElementById('badge-arena').textContent   = 'ARENA_MAX=' + arena;
    document.getElementById('badge-bfc').textContent     = 'TF_BFC=' + bfc;
    document.getElementById('badge-bfc').className       = 'badge ' + (bfc === '0' ? 'ok' : 'warn');
    document.getElementById('badge-tcmalloc').textContent = tcm ? 'tcmalloc ✓' : 'tcmalloc ✗';
    document.getElementById('badge-tcmalloc').className   = 'badge ' + (tcm ? 'ok' : '');

    // 헤더 통계
    document.getElementById('stats').textContent =
      `iter=${j.iterations}  rss=${j.rss_mb.toFixed(1)} MB`;

    // 진단 패널
    const d = j.diag || {};
    document.getElementById('d-rss').textContent    = (d.rss_mb    ?? 0).toFixed(1) + ' MB';
    document.getElementById('d-uss').textContent    = (d.uss_mb    ?? 0).toFixed(1) + ' MB';
    document.getElementById('d-py').textContent     = (d.py_tracemalloc_cur_mb ?? 0).toFixed(1) + ' MB';
    document.getElementById('d-native').textContent = (d.native_est_mb ?? 0).toFixed(1) + ' MB';
    document.getElementById('d-gc').textContent     = (d.gc_objects ?? 0).toLocaleString();
    document.getElementById('d-iter').textContent   = j.iterations;

    if (j.error) document.getElementById('err').textContent = j.error;

    // 메인 차트 업데이트
    if (j.series && j.series.length) {
      const s = j.series;
      mainChart.data.labels                   = s.map(([t]) => Number(t).toFixed(1));
      mainChart.data.datasets[0].data         = s.map(([,r])   => (r / 1024 / 1024).toFixed(2));
      mainChart.data.datasets[1].data         = s.map(([,,p])  => (p / 1024 / 1024).toFixed(2));
      mainChart.data.datasets[2].data         = s.map(([,,,n]) => (n / 1024 / 1024).toFixed(2));
      mainChart.update('none');

      // 증가율 차트
      if (s.length !== prevSeries.length) {
        const rates = computeRates(s);
        rateChart.data.labels            = rates.labels;
        rateChart.data.datasets[0].data  = rates.rss;
        rateChart.data.datasets[1].data  = rates.native;
        rateChart.update('none');
        prevSeries = s;
      }
    }
  } catch(e) {
    document.getElementById('err').textContent = String(e);
  }
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""


# ── FastAPI app ───────────────────────────────────────────────────────────────
def create_app(cfg: RunConfig, tf_version: str, tcmalloc_loaded: bool) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t_inf = threading.Thread(target=_inference_loop, args=(cfg,), daemon=True, name="tf-infer")
        t_log = threading.Thread(target=_rss_logger,     args=(cfg.log_prefix,), daemon=True, name="rss-log")
        t_inf.start()
        t_log.start()
        try:
            yield
        finally:
            _stop_event.set()

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html_page()

    @app.get("/api/memory")
    def api_memory():
        with _state_lock:
            series = list(_rss_history)
            err    = _worker_error
            it     = _iter_count
        proc   = psutil.Process(os.getpid())
        rss    = proc.memory_info().rss
        return JSONResponse({
            "rss_bytes":   rss,
            "rss_mb":      rss / 1024 / 1024,
            "iterations":  it,
            "series":      series,
            "error":       err,
            "diag":        _latest_diag,
            "tensorflow_version": tf_version,
            "config": {
                "malloc_arena_max":    os.environ.get("MALLOC_ARENA_MAX"),
                "tf_allocator_use_bfc": os.environ.get("TF_ALLOCATOR_USE_BFC"),
                "tcmalloc_release_rate": os.environ.get("TCMALLOC_RELEASE_RATE"),
                "tcmalloc_loaded":     tcmalloc_loaded,
                "warmup_steps":        cfg.warmup_steps,
                "gc_every":            cfg.gc_every,
                "clear_every":         cfg.clear_every,
                "engine":              "tf_fn",
            },
        })

    return app


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # 환경변수 기본값 (import 전 설정이 이상적이나 argparse 이전에도 적용)
    os.environ.setdefault("MALLOC_ARENA_MAX",   "2")
    os.environ.setdefault("TF_ALLOCATOR_USE_BFC", "0")
    os.environ.setdefault("TCMALLOC_RELEASE_RATE", "10")

    if not tracemalloc.is_tracing():
        tracemalloc.start(25)

    p = argparse.ArgumentParser(
        description="sixth_try: tf_fn + pre-warm + malloc_trim plateau harness"
    )
    p.add_argument("--opt1", action="store_true", help="Wider model (+32 Conv1D channels)")
    p.add_argument("--opt2", action="store_true", help="+1 extra infer call per segment")
    p.add_argument("--opt3", action="store_true", help="+1 more infer call per segment")
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS,
                   help=f"Pre-warm 호출 횟수 (기본 {WARMUP_STEPS})")
    p.add_argument("--gc-every", type=int, default=GC_EVERY,
                   help=f"N iter마다 gc.collect+malloc_trim (기본 {GC_EVERY})")
    args = p.parse_args()

    # tcmalloc 로드 시도 (LD_PRELOAD 없을 때 fallback)
    tcmalloc_loaded = _try_load_tcmalloc()
    if not tcmalloc_loaded:
        print("[sixth_try] tcmalloc not found — falling back to glibc+MALLOC_ARENA_MAX=2", flush=True)
        print("[sixth_try]   (선택 사항: sudo apt install -y google-perftools 후 재실행하면 tcmalloc 활성화)", flush=True)

    import tensorflow as tf
    import uvicorn

    cfg = RunConfig(
        warmup_steps = args.warmup_steps,
        gc_every     = args.gc_every,
        opt1         = args.opt1,
        opt2         = args.opt2,
        opt3         = args.opt3,
    )
    tf_version = f"tensorflow {tf.__version__}"
    app = create_app(cfg, tf_version, tcmalloc_loaded)

    print(f"[sixth_try] Web UI → http://127.0.0.1:{WEB_PORT}/", flush=True)
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="warning")


if __name__ == "__main__":
    main()
