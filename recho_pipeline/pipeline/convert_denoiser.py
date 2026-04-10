"""
TFLite INT8 conversion for the Hopf-RPU denoiser.

This mirrors the existing converter but targets the denoising checkpoint and
stores firmware artefacts in `firmware/denoiser/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.denoise_data import generate_synthetic_paired_dataset
from pipeline.convert import (
    convert_to_tflite_int8,
    extract_cmsis_nn_params,
    generate_cmsis_nn_header,
    generate_model_data_cc,
    print_deployment_summary,
)
from pipeline.denoise_ingest import prepare_denoising_dataset
from pipeline.models.denoising.tcn_denoiser import TCNDenoiser, representative_data_gen


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the TCN denoiser to TFLite INT8.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/denoiser_tcn.keras")
    parser.add_argument("--firmware_dir", type=str, default="firmware/denoiser")
    parser.add_argument("--n_calib", type=int, default=24)
    args = parser.parse_args()

    model = TCNDenoiser.load(ROOT / args.checkpoint)
    input_timesteps = int(model._model.input_shape[1])
    duration_s = input_timesteps / 4000.0
    clean, _, mixture = generate_synthetic_paired_dataset(
        n_clips=args.n_calib,
        duration_s=duration_s,
        seed=77,
    )
    noisy_inputs, _ = prepare_denoising_dataset(mixture, clean)

    fw_dir = ROOT / args.firmware_dir
    fw_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = fw_dir / "model.tflite"

    def rep_gen():
        return representative_data_gen(noisy_inputs, n_samples=args.n_calib)

    convert_to_tflite_int8(model._model, rep_gen, tflite_path)
    layer_params = extract_cmsis_nn_params(tflite_path)
    generate_cmsis_nn_header(layer_params, fw_dir / "cmsis_nn_params.h")
    generate_model_data_cc(tflite_path, fw_dir / "model_data.cc")
    print_deployment_summary(tflite_path)


if __name__ == "__main__":
    main()
