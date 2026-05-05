#!/usr/bin/env python3
"""
fifth_try/app.py — memory-root-cause isolation + #58676 workaround test harness.

Mirrors IPM_SEGMENTATION_PROCESS.segment() + IPM_SEGMENTATION.predict_batch():
  - Variable batch 1..N_PATIENTS, ROI + filters, 0.2s poll
  - Every CLEAR_EVERY iters: K.clear_session() + gc.collect()
  - **No** model reload after clear_session (same as upstream — optional rebuild removed)

Optional stress (stronger TF / faster RSS growth): ``--opt1`` (wider model), ``--opt2`` / ``--opt3`` (extra ``predict_on_batch`` per segment). Combine as needed.

RSS logged every second to <LOG_PREFIX>.jsonl and .csv.

Live RSS trend: FastAPI + Chart.js on ``FIFTH_TRY_WEB_HOST`` / ``FIFTH_TRY_WEB_PORT`` (defaults ``0.0.0.0:8765``), same pattern as ``second_try``.

``run.sh`` sets ``MALLOC_ARENA_MAX=2`` by default; ``main()`` also applies that default if the variable is unset so ``python app.py`` matches.
"""
from __future__ import annotations

import argparse
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
from typing import Deque, Literal, Optional

import numpy as np
import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# ── Constants matching upstream SEGMENTATION_PARAM ────────────────────────────
ECG_INPUT_NUM = 768       # SEGMENTATION_PARAM['INPUT_NUM'] = 256 * 3
ECG_SAMPLE_RATE = 256
ECG_BUFFER_LEN = 20_000   # per-patient circular ECG buffer (samples)
N_PATIENTS = 50
N_CLASSES = 6

# ── Default run profile (override with ``--opt1`` / ``--opt2`` / ``--opt3``) ─
LOG_PREFIX = "rss_log"
BASE_CH = 32
POLL_SLEEP = 0.2
CLEAR_EVERY = 50
MAX_PATIENTS = N_PATIENTS

# opt1: add this many channels to every Conv1D width step (heavier graph + activations).
BASE_CH_STRESS = 32

WEB_HOST = os.environ.get("FIFTH_TRY_WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("FIFTH_TRY_WEB_PORT", "8765"))


@dataclass(frozen=True)
class RunConfig:
    log_prefix: str = LOG_PREFIX
    base_ch: int = BASE_CH
    poll_sleep: float = POLL_SLEEP
    clear_every: int = CLEAR_EVERY
    max_patients: int = MAX_PATIENTS
    opt1: bool = False  # wider model (+channels)
    opt2: bool = False  # +1 extra predict_on_batch per segment
    opt3: bool = False  # +1 more predict_on_batch per segment
    engine: Literal["predict", "tf_fn"] = "tf_fn"
    # tf_fn mode keeps a single traced function; useful for checking TF #58676-like growth.


def _effective_base_ch(cfg: RunConfig) -> int:
    return cfg.base_ch + (BASE_CH_STRESS if cfg.opt1 else 0)


def _extra_predict_passes(cfg: RunConfig) -> int:
    """Additional ``predict_on_batch`` calls after the first (opt2 and opt3 each add one)."""
    return int(cfg.opt2) + int(cfg.opt3)


# ── Global state ──────────────────────────────────────────────────────────────
_iter_count = 0
_state_lock = threading.Lock()
_stop_event = threading.Event()
# Cumulative samples for the chart: (elapsed_sec_since_infer_start, rss_bytes). No maxlen — full run from t≈0.
_rss_history: Deque[tuple[float, float]] = deque()
_worker_error: str | None = None
_latest_diag: dict[str, float | int | str] = {}


# ── Synthetic ECG filter (mirrors SEERS_ECG_TOOLS.apply_filters) ─────────────


def _diag_snapshot(proc: psutil.Process) -> dict[str, float | int | str]:
    """Best-effort split: Python-managed vs native/TF allocator memory."""
    rss = float(proc.memory_info().rss)
    uss = 0.0
    try:
        uss = float(proc.memory_full_info().uss)
    except Exception:
        uss = 0.0

    py_cur = py_peak = 0.0
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        py_cur = float(cur)
        py_peak = float(peak)

    native_est = max(0.0, rss - py_cur)
    gc_objs = len(gc.get_objects())
    allocated_blocks = sys.getallocatedblocks()

    return {
        "rss_mb": rss / (1024 * 1024),
        "uss_mb": uss / (1024 * 1024),
        "py_tracemalloc_cur_mb": py_cur / (1024 * 1024),
        "py_tracemalloc_peak_mb": py_peak / (1024 * 1024),
        "native_est_mb": native_est / (1024 * 1024),
        "gc_objects": gc_objs,
        "allocated_blocks": allocated_blocks,
    }


def _classify_growth(diag: dict[str, float | int | str]) -> str:
    py_mb = float(diag.get("py_tracemalloc_cur_mb", 0.0))
    native_mb = float(diag.get("native_est_mb", 0.0))
    if native_mb > py_mb * 2.0 and native_mb > 200:
        return "likely_native_or_tf_allocator_growth"
    if py_mb > native_mb and py_mb > 100:
        return "likely_python_heap_growth"
    return "mixed_or_inconclusive"

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
        fc.write("ts_unix,elapsed_s,rss_mb,py_cur_mb,native_est_mb,iter\n")
        while not _stop_event.is_set():
            ts = time.time()
            diag = _diag_snapshot(proc)
            rss_mb = float(diag["rss_mb"])
            py_cur_mb = float(diag["py_tracemalloc_cur_mb"])
            native_est_mb = float(diag["native_est_mb"])
            with _state_lock:
                it = _iter_count
                _latest_diag.update(diag)
            elapsed = ts - t0
            row = {
                "ts_unix": round(ts, 3),
                "elapsed_s": round(elapsed, 1),
                "rss_mb": round(rss_mb, 2),
                "py_cur_mb": round(py_cur_mb, 2),
                "native_est_mb": round(native_est_mb, 2),
                "iter": it,
            }
            fj.write(json.dumps(row) + "\n")
            fc.write(f"{ts:.3f},{elapsed:.1f},{rss_mb:.2f},{py_cur_mb:.2f},{native_est_mb:.2f},{it}\n")
            fj.flush()
            fc.flush()
            time.sleep(1.0)


# ── segment() — mirrors IPM_SEGMENTATION_PROCESS.segment ─────────────────────
def _segment(
    model,
    infer_fn,
    container_manager: FakeContainerManager,
    needed_ids: list,
    rng: np.random.Generator,
    cfg: RunConfig,
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
            if cfg.engine == "predict":
                _predict = model.predict_on_batch(tf_arr)
                for _ in range(_extra_predict_passes(cfg)):
                    _predict = model.predict_on_batch(tf_arr)
            else:
                _predict = infer_fn(tf_arr)
                for _ in range(_extra_predict_passes(cfg)):
                    _predict = infer_fn(tf_arr)
            del tf_arr

        predicted_segments = np.array(_predict, copy=True)

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
    del predicted_segments
    del needed_ids


# ── Inference loop ────────────────────────────────────────────────────────────
def _inference_loop(cfg: RunConfig) -> None:
    global _iter_count, _worker_error

    import tensorflow as tf
    from tensorflow.keras import backend as K

    try:
        K.clear_session()
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        eff_ch = _effective_base_ch(cfg)
        n_pred = 1 + _extra_predict_passes(cfg)
        model = _build_ecg_segmentation_model(ECG_INPUT_NUM, eff_ch)
        model.compile(optimizer="adam", loss="mse")

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, ECG_INPUT_NUM, 1], dtype=tf.float64)])
        def infer_fn(x):
            return model(x, training=False)

        rng = np.random.default_rng(42)
        container_manager = FakeContainerManager(N_PATIENTS, rng)
        proc = psutil.Process(os.getpid())

        print("[fifth_try] Inference loop (#58676 diagnosis harness)")
        print(
            f"  stress opts    = opt1={cfg.opt1} opt2={cfg.opt2} opt3={cfg.opt3}  "
            f"→ base_ch {cfg.base_ch}→{eff_ch}  predict_calls×{n_pred}/seg  engine={cfg.engine}"
        )
        print(f"  variable_batch = True  max_patients={cfg.max_patients}")
        print(f"  use_roi        = True")
        print(f"  clear_session  = True  clear_every={cfg.clear_every}")
        print(f"  rebuild_after_clear = False  (model NOT reloaded after clear)")
        print("  #58676 workaround path: engine=tf_fn uses one traced tf.function")
        print(f"  poll_sleep     ={cfg.poll_sleep}s")
        print(f"  MALLOC_ARENA_MAX={os.environ.get('MALLOC_ARENA_MAX', '(not set)')}")
        print(f"  log            ={cfg.log_prefix}.jsonl / .csv")
        print(f"  web            =http://127.0.0.1:{WEB_PORT}/")
        sys.stdout.flush()

        loop_t0 = time.perf_counter()
        with _state_lock:
            _rss_history.clear()
            _rss_history.append((0.0, float(proc.memory_info().rss)))

        iter_cnt = 0
        t_start = time.time()
        t_last_print = t_start

        while not _stop_event.is_set():
            container_manager.stream_advance(50)

            needed_ids = container_manager.active_ids(rng, max_n=cfg.max_patients)

            _segment(
                model=model,
                infer_fn=infer_fn,
                container_manager=container_manager,
                needed_ids=needed_ids,
                rng=rng,
                cfg=cfg,
            )

            iter_cnt += 1
            rss = float(proc.memory_info().rss)
            with _state_lock:
                _iter_count = iter_cnt
                _rss_history.append((time.perf_counter() - loop_t0, rss))

            # Upstream: clear_session only — do not reload model (leaves stale Python ref;
            # reproduces production behaviour / RSS spikes + trace-cache accumulation).
            if iter_cnt % cfg.clear_every == 0:
                K.clear_session()
                gc.collect()

            now = time.time()
            if now - t_last_print >= 10.0:
                rss_mb = rss / (1024 * 1024)
                print(
                    f"[{now - t_start:6.0f}s] iter={iter_cnt:6d}  "
                    f"rss={rss_mb:.1f} MB  batch={len(needed_ids)}"
                )
                sys.stdout.flush()
                t_last_print = now

            time.sleep(cfg.poll_sleep)
    except Exception as e:  # noqa: BLE001
        with _state_lock:
            _worker_error = repr(e)


def _html_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RSS trend — fifth_try (#58676 diagnosis)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #0f1419; color: #e6edf3; }
    header { padding: 1rem 1.25rem; border-bottom: 1px solid #30363d; }
    h1 { font-size: 1.1rem; font-weight: 600; margin: 0; }
    .meta { font-size: 0.85rem; color: #8b949e; margin-top: 0.35rem; }
    main { padding: 1rem 1.25rem; max-width: 1100px; margin: 0 auto; }
    .chart-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; height: 420px; }
    .error { color: #f85149; margin-top: 1rem; white-space: pre-wrap; }
  </style>
</head>
<body>
  <header>
    <h1>RSS (resident set) — fifth_try root-cause diagnosis</h1>
    <p class="meta" id="meta">Loading…</p>
  </header>
  <main>
    <div class="chart-wrap"><canvas id="rss"></canvas></div>
    <p class="error" id="err"></p>
  </main>
  <script>
    const ctx = document.getElementById('rss');
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'RSS (MB)',
          data: [],
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88,166,255,0.08)',
          fill: true,
          tension: 0.15,
          pointRadius: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            ticks: { color: '#8b949e', maxTicksLimit: 14 },
            title: { display: true, text: 'Elapsed (s)', color: '#8b949e' }
          },
          y: {
            ticks: { color: '#8b949e' },
            title: { display: true, text: 'MB', color: '#8b949e' }
          }
        },
        plugins: { legend: { labels: { color: '#e6edf3' } } }
      }
    });

    async function tick() {
      try {
        const r = await fetch('/api/memory');
        const j = await r.json();
        let extra = '';
        if (j.load) {
          extra = '  load=' + JSON.stringify(j.load);
        }
        document.getElementById('meta').textContent =
          'iter=' + j.iterations + '  rss_mb=' + j.rss_mb.toFixed(2) +
          '  backend=' + j.tensorflow_version + '  classify=' + (j.growth_class || 'n/a') + extra;
        if (j.error) {
          document.getElementById('err').textContent = j.error;
        } else {
          document.getElementById('err').textContent = '';
        }
        if (j.series && j.series.length) {
          chart.data.labels = j.series.map(([sec]) => Number(sec).toFixed(1));
          chart.data.datasets[0].data = j.series.map(([, rss]) => rss / (1024 * 1024));
          chart.update('none');
        }
      } catch (e) {
        document.getElementById('err').textContent = String(e);
      }
    }
    setInterval(tick, 500);
    tick();
  </script>
</body>
</html>
"""


def create_app(cfg: RunConfig, tf_version_label: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t_inf = threading.Thread(
            target=_inference_loop, args=(cfg,), daemon=True, name="tf-infer"
        )
        t_log = threading.Thread(
            target=_rss_logger, args=(cfg.log_prefix,), daemon=True, name="rss-log"
        )
        t_inf.start()
        t_log.start()
        try:
            yield
        finally:
            _stop_event.set()

    app = FastAPI(lifespan=lifespan)
    app.state.cfg = cfg
    app.state.tf_version_label = tf_version_label

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html_page()

    @app.get("/api/memory")
    def api_memory():
        with _state_lock:
            series = list(_rss_history)
            err = _worker_error
            it = _iter_count
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        c: RunConfig = app.state.cfg
        eff = _effective_base_ch(c)
        load = {
            "fifth_try": True,
            "opt1": c.opt1,
            "opt2": c.opt2,
            "opt3": c.opt3,
            "base_ch_config": c.base_ch,
            "base_ch_effective": eff,
            "predict_on_batch_per_segment": 1 + _extra_predict_passes(c),
            "clear_every": c.clear_every,
            "poll_sleep_sec": c.poll_sleep,
            "max_patients": c.max_patients,
            "ecg_input_num": ECG_INPUT_NUM,
            "log_prefix": c.log_prefix,
            "engine": c.engine,
        }
        return JSONResponse(
            {
                "rss_bytes": rss,
                "rss_mb": rss / (1024 * 1024),
                "iterations": it,
                "leak": False,
                "ipm_predict_batch_like": True,
                "tensorflow_version": app.state.tf_version_label,
                "load": load,
                "series": series,
                "error": err,
                "diag": _latest_diag,
                "growth_class": _classify_growth(_latest_diag),
            }
        )

    return app


def main() -> None:
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)

    p = argparse.ArgumentParser(
        description="fifth_try memory diagnosis harness. Compare Python heap vs TF/native growth and test #58676 workaround path."
    )
    p.add_argument(
        "--opt1",
        action="store_true",
        help=f"Stress: wider model (+{BASE_CH_STRESS} to base Conv1D channels).",
    )
    p.add_argument(
        "--opt2",
        action="store_true",
        help="Stress: +1 extra predict_on_batch per segment (after the first).",
    )
    p.add_argument(
        "--opt3",
        action="store_true",
        help="Stress: +1 more predict_on_batch per segment (cumulative with --opt2).",
    )
    p.add_argument(
        "--engine",
        choices=["predict", "tf_fn"],
        default="tf_fn",
        help="predict=model.predict_on_batch path, tf_fn=single traced model(x) path (workaround candidate).",
    )
    args = p.parse_args()

    import tensorflow as tf
    import uvicorn

    cfg = RunConfig(
        log_prefix=LOG_PREFIX,
        base_ch=BASE_CH,
        poll_sleep=POLL_SLEEP,
        clear_every=CLEAR_EVERY,
        max_patients=MAX_PATIENTS,
        opt1=args.opt1,
        opt2=args.opt2,
        opt3=args.opt3,
        engine=args.engine,
    )
    tf_version_label = f"tensorflow {tf.__version__}"
    app = create_app(cfg, tf_version_label)
    print(
        f"[fifth_try] Web UI http://127.0.0.1:{WEB_PORT}/  (bind {WEB_HOST}:{WEB_PORT})",
        flush=True,
    )
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="info")


if __name__ == "__main__":
    main()
