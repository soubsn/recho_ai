"""
Training entrypoint for Hopf-RPU denoising.

This workflow is intentionally separate from `train_all.py`. It trains one
causal sequence-to-sequence denoiser on paired noisy/clean waveforms after the
noisy mixture has been transformed by the Hopf reservoir.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.denoise_data import generate_synthetic_paired_dataset, train_val_test_split
from pipeline.denoise_ingest import prepare_denoising_dataset
from pipeline.models.denoising.tcn_denoiser import TCNDenoiser, si_sdr_db_numpy, snr_db_numpy


def _save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    n_epochs = len(history[keys[0]]) if keys else 0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + keys)
        writer.writeheader()
        for epoch in range(n_epochs):
            row = {"epoch": epoch + 1}
            for key in keys:
                row[key] = history[key][epoch]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Hopf-RPU TCN denoiser.")
    parser.add_argument("--n_clips", type=int, default=120)
    parser.add_argument("--duration_s", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="output/denoising")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/denoiser_tcn.keras")
    args = parser.parse_args()

    print("[train_denoiser] Generating paired training data ...")
    clean, noise, mixture = generate_synthetic_paired_dataset(
        n_clips=args.n_clips,
        duration_s=args.duration_s,
        seed=0,
    )
    splits = train_val_test_split(clean, noise, mixture, val_fraction=0.15, test_fraction=0.15, seed=0)

    print("[train_denoiser] Preparing reservoir sequences ...")
    x_train, y_train = prepare_denoising_dataset(splits["train"]["mixture"], splits["train"]["clean"])
    x_val, y_val = prepare_denoising_dataset(splits["val"]["mixture"], splits["val"]["clean"])
    x_test, y_test = prepare_denoising_dataset(splits["test"]["mixture"], splits["test"]["clean"])

    model = TCNDenoiser(epochs=args.epochs, batch_size=args.batch_size)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), checkpoint_path=args.checkpoint)

    preds = model.predict(x_test)
    baseline_snr = float(snr_db_numpy(y_test, splits["test"]["mixture"][:, :, None]).mean())
    denoised_snr = float(snr_db_numpy(y_test, preds).mean())
    denoised_si_sdr = float(si_sdr_db_numpy(y_test, preds).mean())

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_history_csv(model.history_, out_dir / "training_history.csv")

    print("[train_denoiser] Done")
    print(f"  train inputs: {x_train.shape}")
    print(f"  val inputs:   {x_val.shape}")
    print(f"  test inputs:  {x_test.shape}")
    print(f"  baseline SNR: {baseline_snr:.3f} dB")
    print(f"  denoised SNR: {denoised_snr:.3f} dB")
    print(f"  denoised SI-SDR: {denoised_si_sdr:.3f} dB")
    print(f"  history: {out_dir / 'training_history.csv'}")
    print(f"  checkpoint: {Path(args.checkpoint)}")


if __name__ == "__main__":
    main()
