#!/usr/bin/env python3
"""
third_try/app.py — upstream-faithful memory leak reproduction (single fixed profile).

Mirrors IPM_SEGMENTATION_PROCESS.segment() + IPM_SEGMENTATION.predict_batch():
  - Variable batch 1..N_PATIENTS, ROI + filters, 0.2s poll
  - Every CLEAR_EVERY iters: K.clear_session() + gc.collect()
  - **No** model reload after clear_session (same as upstream — optional rebuild removed)

RSS logged every second to <LOG_PREFIX>.jsonl and .csv.
"""
from __future__ import annotations

import gc
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import psutil

# ── Constants matching upstream SEGMENTATION_PARAM ────────────────────────────
ECG_INPUT_NUM = 768       # SEGMENTATION_PARAM['INPUT_NUM'] = 256 * 3
ECG_SAMPLE_RATE = 256
ECG_BUFFER_LEN = 20_000   # per-patient circular ECG buffer (samples)
N_PATIENTS = 50
N_CLASSES = 6

# ── Single run profile (no CLI — edit here if you need to tune) ─────────────
LOG_PREFIX = "rss_log"
BASE_CH = 32
POLL_SLEEP = 0.2
CLEAR_EVERY = 1000
MAX_PATIENTS = N_PATIENTS


@dataclass(frozen=True)
class RunConfig:
    log_prefix: str = LOG_PREFIX
    base_ch: int = BASE_CH
    poll_sleep: float = POLL_SLEEP
    clear_every: int = CLEAR_EVERY
    max_patients: int = MAX_PATIENTS


# ── Global state ──────────────────────────────────────────────────────────────
_iter_count = 0
_state_lock = threading.Lock()
_stop_event = threading.Event()


# ── Synthetic ECG filter (mirrors SEERS_ECG_TOOLS.apply_filters) ─────────────
def _apply_filters(ecg_roi: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """Bandpass-like filter using numpy — allocates temporary buffers like scipy."""
    if ecg_roi is None or len(ecg_roi) < 10:
        return None
    sig = ecg_roi.astype(np.float64)
    k = max(3, sample_rate // 50)
    kernel = np.ones(k, dtype=np.float64) / k
    lp = np.convolve(sig, kernel, mode="same")
    lp -= np.mean(lp)
    return lp


# ── Fake patient / container (mirrors CONTAINER_MANAGER) ─────────────────────
class _FakePatient:
    """One patient's streaming ECG circular buffer."""

    def __init__(self, pid: int, rng: np.random.Generator) -> None:
        self.pid = pid
        t = np.linspace(0, ECG_BUFFER_LEN / ECG_SAMPLE_RATE, ECG_BUFFER_LEN)
        hr_hz = rng.uniform(50, 100) / 60.0
        buf = (
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
        buf = self.ecg_buffer
        buf_len = len(buf)
        length = end - start
        if length <= 0:
            return np.zeros(ECG_INPUT_NUM, dtype=np.float64)
        sc = start % buf_len
        ec = end % buf_len
        if ec > sc:
            roi = buf[sc:ec]
        elif ec == sc:
            roi = buf[sc : sc + length]
        else:
            roi = np.concatenate([buf[sc:], buf[:ec]])
        if len(roi) < length:
            roi = np.pad(roi, (0, length - len(roi)))
        return roi[:length].copy()


class FakeContainerManager:
    """Mirrors CONTAINER_MANAGER.get / get_roi / set interface."""

    def __init__(self, n_patients: int, rng: np.random.Generator) -> None:
        self._patients: dict = {i: _FakePatient(i, rng) for i in range(n_patients)}
        self._results: dict = {}
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
        start, end = index_range
        return p.get_roi(start, end)

    def set(self, seg_data: dict, param: str) -> None:
        with self._lock:
            self._results[seg_data.get("patient_id", 0)] = seg_data

    def active_ids(self, rng: np.random.Generator, max_n: int) -> list:
        with self._lock:
            all_ids = list(self._patients.keys())
        n = int(rng.integers(1, min(max_n, len(all_ids)) + 1))
        chosen = rng.choice(len(all_ids), size=n, replace=False)
        return [all_ids[i] for i in chosen]

    def stream_advance(self, n: int = 50) -> None:
        with self._lock:
            patients = list(self._patients.values())
        for p in patients:
            p.advance(n)


# ── Model builder ─────────────────────────────────────────────────────────────
def _build_ecg_segmentation_model(ecg_length: int, base_ch: int):
    """1-D ECG segmentation model (mirrors real IPM model shape)."""
    import tensorflow as tf

    inp = tf.keras.Input(shape=(ecg_length, 1), name="ecg_signal")
    x = tf.keras.layers.Conv1D(base_ch, 7, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(base_ch * 2, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(base_ch * 2, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(base_ch, 3, padding="same", activation="relu")(x)
    out = tf.keras.layers.Conv1D(
        N_CLASSES, 1, padding="same", activation="linear", name="logits"
    )(x)
    return tf.keras.Model(inp, out, name="ecg_seg_dummy")


# ── RSS logger ────────────────────────────────────────────────────────────────
def _rss_logger(log_prefix: str) -> None:
    proc = psutil.Process(os.getpid())
    t0 = time.time()
    with open(log_prefix + ".jsonl", "w", encoding="utf-8") as fj, open(
        log_prefix + ".csv", "w", encoding="utf-8"
    ) as fc:
        fc.write("ts_unix,elapsed_s,rss_mb,iter\n")
        while not _stop_event.is_set():
            ts = time.time()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            with _state_lock:
                it = _iter_count
            elapsed = ts - t0
            row = {
                "ts_unix": round(ts, 3),
                "elapsed_s": round(elapsed, 1),
                "rss_mb": round(rss_mb, 2),
                "iter": it,
            }
            fj.write(json.dumps(row) + "\n")
            fc.write(f"{ts:.3f},{elapsed:.1f},{rss_mb:.2f},{it}\n")
            fj.flush()
            fc.flush()
            time.sleep(1.0)


# ── segment() — mirrors IPM_SEGMENTATION_PROCESS.segment ─────────────────────
def _segment(
    model,
    container_manager: FakeContainerManager,
    needed_ids: list,
    rng: np.random.Generator,
) -> None:
    import tensorflow as tf

    ids = []
    edIdxs = []
    FEcgROIs = []
    signals = []
    predicted_segments = []

    for id in needed_ids:
        try:
            indexes = container_manager.get(id, "ECGINDEX")
            if indexes is None:
                continue

            _, edIdx, _, _, _ = indexes[0]

            EcgROI = container_manager.get_roi(
                id,
                (edIdx - ECG_INPUT_NUM - 1, edIdx - 1),
                "ECGVOLT",
            )
            FEcgROI = _apply_filters(EcgROI, ECG_SAMPLE_RATE)

            if FEcgROI is not None:
                ids.append(id)
                edIdxs.append(edIdx)
                FEcgROIs.append(FEcgROI)

            del id
            del edIdx
            del EcgROI
            del FEcgROI
            del indexes

        except Exception:
            continue

    if not FEcgROIs:
        del ids, edIdxs, FEcgROIs, signals, predicted_segments, needed_ids
        return

    try:
        signals = np.array(FEcgROIs)
        signals2 = signals.reshape(signals.shape[0], signals.shape[1], 1)
        del signals
        signals = signals2

        with tf.device("/CPU:0"):
            tf_arr = tf.convert_to_tensor(signals, dtype=tf.float64)
            _predict = model.predict_on_batch(tf_arr)
            del tf_arr

        predicted_segments = _predict.copy()

        del signals, signals2

    except Exception:
        del ids, edIdxs, FEcgROIs
        return

    for id, segment, edIdx in zip(ids, predicted_segments, edIdxs):
        try:
            pqrst = np.argmax(segment, axis=1)
            seg_data = {"patient_id": id, "pqrst": pqrst, "edIdx": edIdx}
            container_manager.set(seg_data, "SEGMENTATION")

            del id
            del segment
            del edIdx
            del pqrst
            del seg_data
        except Exception:
            continue

    del ids
    del edIdxs
    del FEcgROIs
    del signals
    del predicted_segments
    del needed_ids


# ── Inference loop ────────────────────────────────────────────────────────────
def _inference_loop(cfg: RunConfig) -> None:
    global _iter_count

    import tensorflow as tf
    from tensorflow.keras import backend as K

    K.clear_session()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model = _build_ecg_segmentation_model(ECG_INPUT_NUM, cfg.base_ch)
    model.compile(optimizer="adam", loss="mse")

    rng = np.random.default_rng(42)
    container_manager = FakeContainerManager(N_PATIENTS, rng)
    proc = psutil.Process(os.getpid())

    print("[third_try] Inference loop (upstream-faithful, no CLI)")
    print(f"  variable_batch = True  max_patients={cfg.max_patients}")
    print(f"  use_roi        = True")
    print(f"  clear_session  = True  clear_every={cfg.clear_every}")
    print(f"  rebuild_after_clear = False  (model NOT reloaded after clear)")
    print(f"  poll_sleep     ={cfg.poll_sleep}s")
    print(f"  MALLOC_ARENA_MAX={os.environ.get('MALLOC_ARENA_MAX', '(not set)')}")
    print(f"  log            ={cfg.log_prefix}.jsonl / .csv")
    sys.stdout.flush()

    iter_cnt = 0
    t_start = time.time()
    t_last_print = t_start

    while not _stop_event.is_set():
        container_manager.stream_advance(50)

        needed_ids = container_manager.active_ids(rng, max_n=cfg.max_patients)

        _segment(
            model=model,
            container_manager=container_manager,
            needed_ids=needed_ids,
            rng=rng,
        )

        iter_cnt += 1
        with _state_lock:
            _iter_count = iter_cnt

        # Upstream: clear_session only — do not reload model (leaves stale Python ref;
        # reproduces production behaviour / RSS spikes + trace-cache accumulation).
        if iter_cnt % cfg.clear_every == 0:
            K.clear_session()
            gc.collect()

        now = time.time()
        if now - t_last_print >= 10.0:
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            print(
                f"[{now - t_start:6.0f}s] iter={iter_cnt:6d}  "
                f"rss={rss_mb:.1f} MB  batch={len(needed_ids)}"
            )
            sys.stdout.flush()
            t_last_print = now

        time.sleep(cfg.poll_sleep)


def main() -> None:
    cfg = RunConfig(
        log_prefix=LOG_PREFIX,
        base_ch=BASE_CH,
        poll_sleep=POLL_SLEEP,
        clear_every=CLEAR_EVERY,
        max_patients=MAX_PATIENTS,
    )

    def _handle_signal(*_):
        print("\n[third_try] Signal received, stopping...")
        _stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log_thread = threading.Thread(
        target=_rss_logger, args=(cfg.log_prefix,), daemon=True, name="rss-log"
    )
    log_thread.start()

    try:
        _inference_loop(cfg)
    except KeyboardInterrupt:
        _stop_event.set()

    print("[third_try] Done.")


if __name__ == "__main__":
    main()
