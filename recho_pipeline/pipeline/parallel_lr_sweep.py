"""
Run train_CNN.py's LR sweep as N parallel subprocesses.

Each worker runs exactly one LR value via the RECHO_SINGLE_LR env var and
emits a [RESULT] line on stdout. The parent process captures per-worker
logs to output/lr_sweep_logs/ and prints a sorted summary when all
workers exit.

Usage:
    python recho_pipeline/pipeline/parallel_lr_sweep.py
    python recho_pipeline/pipeline/parallel_lr_sweep.py --workers 2
    python recho_pipeline/pipeline/parallel_lr_sweep.py --lrs 1e-4 3e-4

Defaults: 4 workers, LRs (1e-4, 3e-4, 1e-3, 3e-3).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "train_CNN.py"
LOG_DIR = HERE.parent / "output" / "lr_sweep_logs"

DEFAULT_LRS: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
DEFAULT_WORKERS: int = 4

RESULT_RE = re.compile(
    r"\[RESULT\]\s+lr=([0-9.eE+-]+)\s+best_val_acc=([0-9.]+)\s+"
    r"best_val_loss=([0-9.]+)\s+epochs=(\d+)"
)


def _run_one(lr: float, log_dir: Path) -> tuple[float, float | None, float | None, int | None, int]:
    """
    Spawn one train_CNN.py subprocess pinned to `lr` via RECHO_SINGLE_LR.

    Returns (lr, best_val_acc, best_val_loss, epochs, returncode). The three
    result fields are None if the [RESULT] line was not found.
    """
    log_path = log_dir / f"lr_{lr:.2e}.log"
    env = os.environ.copy()
    env["RECHO_SINGLE_LR"] = f"{lr:.6e}"
    # Force unbuffered Python stdout so logs stream live to the file.
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    with log_path.open("w") as logf:
        logf.write(f"# RECHO_SINGLE_LR={env['RECHO_SINGLE_LR']}\n")
        logf.write(f"# target={TARGET}\n")
        logf.write(f"# started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        logf.flush()
        proc = subprocess.run(
            [sys.executable, "-u", str(TARGET)],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0

    # Parse [RESULT] line from the log.
    text = log_path.read_text()
    m = RESULT_RE.search(text)
    if m is None:
        print(
            f"[parallel-sweep] lr={lr:.2e}: no [RESULT] line found "
            f"(exit={proc.returncode}, {elapsed:.0f}s) — see {log_path}"
        )
        return lr, None, None, None, proc.returncode

    acc, loss, epochs = float(m.group(2)), float(m.group(3)), int(m.group(4))
    print(
        f"[parallel-sweep] lr={lr:.2e} done in {elapsed:.0f}s — "
        f"val_acc={acc:.4f} val_loss={loss:.4f} epochs={epochs}"
    )
    return lr, acc, loss, epochs, proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel LR sweep for train_CNN.py")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"parallel workers (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--lrs", type=float, nargs="+", default=list(DEFAULT_LRS),
        help=f"LR values to sweep (default {list(DEFAULT_LRS)})",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[parallel-sweep] target={TARGET}")
    print(f"[parallel-sweep] workers={args.workers}  lrs={args.lrs}")
    print(f"[parallel-sweep] per-worker logs: {LOG_DIR}/lr_<value>.log")
    print(f"[parallel-sweep] tail one live with: tail -f {LOG_DIR}/lr_3.00e-04.log\n")

    t0 = time.time()
    results: list[tuple[float, float | None, float | None, int | None, int]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, lr, LOG_DIR): lr for lr in args.lrs}
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.time() - t0

    # Sort: successes by best_val_acc desc, failures at the bottom.
    def _key(row):
        _, acc, _, _, _ = row
        return (0, -acc) if acc is not None else (1, 0.0)

    results.sort(key=_key)

    print(f"\n--- Parallel LR sweep summary ({elapsed:.0f}s wall-clock) ---")
    print(f"{'peak_lr':>10}  {'best_val_acc':>13}  {'best_val_loss':>14}  "
          f"{'epochs':>7}  {'exit':>5}")
    print("-" * 60)
    for lr, acc, loss, ep, rc in results:
        if acc is None:
            print(f"{lr:>10.2e}  {'FAILED':>13}  {'—':>14}  {'—':>7}  {rc:>5}")
        else:
            print(f"{lr:>10.2e}  {acc:>13.4f}  {loss:>14.4f}  {ep:>7d}  {rc:>5}")

    successes = [r for r in results if r[1] is not None]
    if successes:
        best = successes[0]
        print(
            f"\n[parallel-sweep] Best peak_lr: {best[0]:.2e} "
            f"(val_accuracy={best[1]:.4f}). "
            f"Set PEAK_LR={best[0]:.2e}, TUNE_LR=False, and rerun to pretrain."
        )
    else:
        print("\n[parallel-sweep] All workers failed — check per-worker logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
