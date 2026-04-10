"""
Evaluation entrypoint for Hopf-RPU denoising.

Reports before/after denoising quality on synthetic paired data and saves a
small visual report under `output/denoising/`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.denoise_data import generate_synthetic_paired_dataset
from pipeline.denoise_ingest import prepare_denoising_dataset
from pipeline.models.denoising.tcn_denoiser import TCNDenoiser, receptive_field, si_sdr_db_numpy, snr_db_numpy


def _save_metrics_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_examples(
    noisy: np.ndarray,
    clean: np.ndarray,
    denoised: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(noisy, color="tab:orange")
    axes[0].set_title("Noisy mixture")
    axes[1].plot(denoised, color="tab:blue")
    axes[1].set_title("Denoised output")
    axes[2].plot(clean, color="tab:green")
    axes[2].set_title("Clean target")
    axes[2].set_xlabel("Time step")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Hopf-RPU denoiser.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/denoiser_tcn.keras")
    parser.add_argument("--n_clips", type=int, default=30)
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="output/denoising")
    args = parser.parse_args()

    model = TCNDenoiser.load(ROOT / args.checkpoint)
    input_timesteps = int(model._model.input_shape[1])
    duration_s = args.duration_s if args.duration_s is not None else input_timesteps / 4000.0
    clean, _, mixture = generate_synthetic_paired_dataset(
        n_clips=args.n_clips,
        duration_s=duration_s,
        seed=123,
    )
    noisy_inputs, clean_targets = prepare_denoising_dataset(mixture, clean)
    preds = model.predict(noisy_inputs)

    baseline_snr = snr_db_numpy(clean_targets, mixture[:, :, None])
    denoised_snr = snr_db_numpy(clean_targets, preds)
    baseline_si_sdr = si_sdr_db_numpy(clean_targets, mixture[:, :, None])
    denoised_si_sdr = si_sdr_db_numpy(clean_targets, preds)

    out_dir = ROOT / args.output_dir
    metrics = [{
        "baseline_snr_db": float(np.mean(baseline_snr)),
        "denoised_snr_db": float(np.mean(denoised_snr)),
        "snr_improvement_db": float(np.mean(denoised_snr - baseline_snr)),
        "baseline_si_sdr_db": float(np.mean(baseline_si_sdr)),
        "denoised_si_sdr_db": float(np.mean(denoised_si_sdr)),
        "si_sdr_improvement_db": float(np.mean(denoised_si_sdr - baseline_si_sdr)),
        "params": float(model._model.count_params()),
        "algorithmic_latency_ms": float((receptive_field() - 1) / 4000.0 * 1000.0),
    }]
    _save_metrics_csv(metrics, out_dir / "evaluation_metrics.csv")
    _plot_examples(
        noisy=mixture[0],
        clean=clean[0],
        denoised=preds[0, :, 0],
        out_path=out_dir / "example_denoising.png",
    )

    print("[evaluate_denoiser] Evaluation complete")
    for key, value in metrics[0].items():
        print(f"  {key}: {value:.4f}")
    print(f"  metrics: {out_dir / 'evaluation_metrics.csv'}")
    print(f"  plot:    {out_dir / 'example_denoising.png'}")


if __name__ == "__main__":
    main()
