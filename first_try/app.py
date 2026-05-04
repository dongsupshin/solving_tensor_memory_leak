#!/usr/bin/env python3
"""
Linux demo: infinite segmentation-style inference + RSS trend API + web UI.

TensorFlow path mirrors ``IPM_SEGMENTATION.predict_batch`` (upstream):
  numpy batch → reshape → del → ``tf.convert_to_tensor(..., float64)``
  → ``predict_on_batch`` → ``del`` → periodic ``K.clear_session()`` + ``gc.collect()``.

Default load is **moderate** so the web UI stays responsive; use flags to push harder.

Tuning: ``--spatial``, ``--batch``, ``--base-ch``, ``--forwards``, ``--clear-every``, ``--yield-forward``, ``--yield-cycle``.

HTTP handlers are **sync** so polling is not blocked by the asyncio loop; ``--yield-forward`` yields GIL to Uvicorn/TF.

run.sh: Python 3.10; TF pins match
``ai/source/ai/upstream/SETTING_SAMPLE/requirements_ubuntu.txt``.
"""
from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, List

import numpy as np
import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# --- shared inference / metrics state (main process = worker + web) ---
_state_lock = threading.Lock()
_iter_count = 0
_rss_history: Deque[tuple[float, float]] = deque(maxlen=600)  # (unix_ts, rss_bytes)
_leak_store: List[np.ndarray] = []
_worker_error: str | None = None


def _build_segmentation_like_model(spatial: int, base_ch: int):
    import tensorflow as tf

    inp = tf.keras.Input(shape=(spatial, spatial, 1), name="ecg_patch")
    x = tf.keras.layers.Conv2D(base_ch, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(base_ch * 2, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(base_ch * 2, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(base_ch * 2, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(base_ch * 2, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(base_ch, 3, padding="same", activation="relu")(x)
    out = tf.keras.layers.Conv2D(6, 1, padding="same", activation="linear", name="logits")(x)
    return tf.keras.Model(inp, out, name="dummy_segmentation")


def _inference_loop_tensorflow(
    leak: bool,
    sleep_s: float,
    *,
    spatial: int,
    batch: int,
    base_ch: int,
    forwards: int,
    clear_every: int,
    yield_forward: float,
    yield_cycle: float,
) -> None:
    """
    Align forward + clear cadence with ``IPM_SEGMENTATION.predict_batch`` (keras mode).
    """
    global _iter_count, _worker_error
    import tensorflow as tf
    from tensorflow.keras import backend as K

    try:
        K.clear_session()
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        model = _build_segmentation_like_model(spatial, base_ch)
        model.compile(optimizer="adam", loss="mse")

        rng = np.random.default_rng(0)
        h = w = spatial
        batch_n = batch
        proc = psutil.Process(os.getpid())
        cycle_i = 0

        with _state_lock:
            _rss_history.append((time.time(), float(proc.memory_info().rss)))

        while True:
            for _ in range(forwards):
                signals = rng.standard_normal((batch_n, h, w), dtype=np.float64)
                signals2 = signals.reshape(batch_n, h, w, 1)
                del signals
                signals = signals2

                tf_arr = tf.convert_to_tensor(signals, dtype=tf.float64)
                _predict = model.predict_on_batch(tf_arr)
                del tf_arr

                out = np.array(_predict, copy=True)
                del signals, signals2

                if leak:
                    _leak_store.append(out)

                with _state_lock:
                    _iter_count += 1
                    rss = proc.memory_info().rss
                    _rss_history.append((time.time(), float(rss)))

                if yield_forward > 0:
                    time.sleep(yield_forward)

            if yield_cycle > 0:
                time.sleep(yield_cycle)

            cycle_i += 1
            if cycle_i % clear_every == 0:
                K.clear_session()
                gc.collect()
                model = _build_segmentation_like_model(spatial, base_ch)
                model.compile(optimizer="adam", loss="mse")
                cycle_i = 0

            if sleep_s > 0:
                time.sleep(sleep_s)
    except Exception as e:  # noqa: BLE001
        with _state_lock:
            _worker_error = repr(e)


def _inference_loop_numpy(
    leak: bool,
    sleep_s: float,
    *,
    spatial: int,
    batch: int,
    forwards: int,
    yield_forward: float,
    yield_cycle: float,
) -> None:
    """Heavier matmul chain scaled by --spatial / --batch / --forwards."""
    global _iter_count, _worker_error
    try:
        rng = np.random.default_rng(0)
        d = spatial * spatial
        hidden = max(256, spatial * 4)
        w1 = rng.standard_normal((8, hidden)).astype(np.float32)
        w2 = rng.standard_normal((hidden, 6)).astype(np.float32)
        proc = psutil.Process(os.getpid())
        with _state_lock:
            _rss_history.append((time.time(), float(proc.memory_info().rss)))
        while True:
            for _ in range(forwards):
                for __ in range(batch):
                    x = rng.standard_normal((d, 8), dtype=np.float32)
                    h = np.maximum(0, x @ w1)
                    logits = h @ w2
                    y = logits.reshape(1, spatial, spatial, 6)
                    del x, h, logits
                    if leak:
                        _leak_store.append(np.array(y, copy=True))
                with _state_lock:
                    _iter_count += 1
                    rss = proc.memory_info().rss
                    _rss_history.append((time.time(), float(rss)))
                if yield_forward > 0:
                    time.sleep(yield_forward)
            if yield_cycle > 0:
                time.sleep(yield_cycle)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except Exception as e:  # noqa: BLE001
        with _state_lock:
            _worker_error = repr(e)


def _html_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RSS trend — segmentation-style loop</title>
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
    <h1>RSS (resident set) — infinite segmentation-style predict</h1>
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
          x: { ticks: { color: '#8b949e', maxTicksLimit: 12 } },
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
          '  leak=' + j.leak + '  backend=' + j.tensorflow_version + extra;
        if (j.error) {
          document.getElementById('err').textContent = j.error;
        }
        if (j.series && j.series.length) {
          const t0 = j.series[0][0];
          chart.data.labels = j.series.map(([ts]) => (ts - t0).toFixed(1));
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


def _resolve_backend(cli_backend: str | None) -> str:
    if cli_backend:
        return cli_backend
    if os.environ.get("NUMPY_DEMO") == "1":
        return "numpy"
    env = os.environ.get("TF_LEAK_BACKEND", "").strip().lower()
    if env in ("numpy", "tensorflow"):
        return env
    return "tensorflow"


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    if args.backend == "tensorflow":
        target = _inference_loop_tensorflow
        name = "tf-infer"
        kwargs = {
            "leak": args.leak,
            "sleep_s": args.sleep,
            "spatial": args.spatial,
            "batch": args.batch,
            "base_ch": args.base_ch,
            "forwards": args.forwards,
            "clear_every": args.clear_every,
            "yield_forward": args.yield_forward,
            "yield_cycle": args.yield_cycle,
        }
    else:
        target = _inference_loop_numpy
        name = "numpy-infer"
        kwargs = {
            "leak": args.leak,
            "sleep_s": args.sleep,
            "spatial": args.spatial,
            "batch": args.batch,
            "forwards": args.forwards,
            "yield_forward": args.yield_forward,
            "yield_cycle": args.yield_cycle,
        }
    t = threading.Thread(target=target, kwargs=kwargs, daemon=True, name=name)
    t.start()
    yield


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.args = args

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Sync route: runs in threadpool so TF infer thread + asyncio are not starved together.
        return _html_page()

    @app.get("/api/memory")
    def api_memory():
        with _state_lock:
            series = list(_rss_history)
            err = _worker_error
            it = _iter_count
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        load = None
        if args.backend == "tensorflow":
            load = {
                "spatial": args.spatial,
                "batch": args.batch,
                "base_ch": args.base_ch,
                "forwards_per_cycle": args.forwards,
                "clear_every_cycles": args.clear_every,
                "yield_forward_sec": args.yield_forward,
                "yield_cycle_sec": args.yield_cycle,
            }
        elif args.backend == "numpy":
            load = {
                "spatial": args.spatial,
                "batch": args.batch,
                "forwards_per_cycle": args.forwards,
                "yield_forward_sec": args.yield_forward,
                "yield_cycle_sec": args.yield_cycle,
            }
        return JSONResponse(
            {
                "rss_bytes": rss,
                "rss_mb": rss / (1024 * 1024),
                "iterations": it,
                "leak": args.leak,
                "ipm_predict_batch_like": args.backend == "tensorflow",
                "tensorflow_version": args.version_label,
                "load": load,
                "series": series,
                "error": err,
            }
        )

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Infinite predict + RSS web monitor")
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=8765, help="HTTP port")
    p.add_argument(
        "--backend",
        choices=["tensorflow", "numpy"],
        default=None,
        help="tensorflow=TF 2.10 (needs AVX). numpy=no TF (VirtualBox-friendly).",
    )
    p.add_argument(
        "--leak",
        action="store_true",
        help="Retain every prediction in RAM (synthetic; not default IPM behavior).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep after each outer cycle (reduce CPU).",
    )
    p.add_argument(
        "--spatial",
        type=int,
        default=96,
        help="H=W input patch size (TF model input). Larger → more RAM / compute.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for predict_on_batch.",
    )
    p.add_argument(
        "--base-ch",
        type=int,
        default=32,
        dest="base_ch",
        help="Base Conv2D channels (scaled x2 in deeper layers).",
    )
    p.add_argument(
        "--forwards",
        type=int,
        default=3,
        help="predict_on_batch calls per outer cycle (before optional sleep).",
    )
    p.add_argument(
        "--yield-forward",
        type=float,
        default=0.006,
        dest="yield_forward",
        help="Sleep after each forward (sec) so /api/memory can respond under GIL+TF load; 0=max throughput.",
    )
    p.add_argument(
        "--yield-cycle",
        type=float,
        default=0.005,
        dest="yield_cycle",
        help="Sleep after each full forwards group (between outer cycles).",
    )
    p.add_argument(
        "--clear-every",
        type=int,
        default=800,
        dest="clear_every",
        help="Run K.clear_session every N outer cycles (TF only). Larger → longer growth between clears.",
    )
    args = p.parse_args()
    args.backend = _resolve_backend(args.backend)

    if args.spatial < 32 or args.spatial > 512:
        p.error("--spatial must be between 32 and 512")
    if args.batch < 1 or args.batch > 128:
        p.error("--batch must be between 1 and 128")
    if args.base_ch < 8 or args.base_ch > 128:
        p.error("--base-ch must be between 8 and 128")
    if args.forwards < 1 or args.forwards > 64:
        p.error("--forwards must be between 1 and 64")
    if args.backend == "tensorflow" and (args.clear_every < 1 or args.clear_every > 100000):
        p.error("--clear-every must be between 1 and 100000")
    if args.yield_forward < 0 or args.yield_forward > 1.0:
        p.error("--yield-forward must be between 0 and 1.0")
    if args.yield_cycle < 0 or args.yield_cycle > 1.0:
        p.error("--yield-cycle must be between 0 and 1.0")

    if args.backend == "tensorflow":
        import tensorflow as tf

        args.version_label = f"tensorflow {tf.__version__}"
    else:
        args.version_label = "numpy-only (TensorFlow not used — for CPUs without AVX / VirtualBox)"

    import uvicorn

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
