"""
TFLite INT8 conversion for the Hopf-RPU denoiser.

This mirrors the existing converter but targets the denoising checkpoint and
stores firmware artefacts in `firmware/denoiser/`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.denoise_data import generate_synthetic_paired_dataset
from pipeline.denoise_ingest import prepare_denoising_dataset
from pipeline.models.denoising.tcn_denoiser import TCNDenoiser, representative_data_gen


def _protobuf_version() -> Tuple[int, ...]:
    import google.protobuf

    version = google.protobuf.__version__.split(".")
    out: list[int] = []
    for part in version:
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            out.append(int(num))
    return tuple(out)


def _ensure_converter_environment() -> None:
    """
    Fail early on protobuf/TFLite combinations known to break Keras conversion.

    TensorFlow 2.16.x is not reliably compatible with protobuf 6 for TFLite
    conversion. In that state the converter crashes with opaque MLIR errors such
    as `MessageFactory.GetPrototype` or `missing attribute 'value'`.
    """
    version = _protobuf_version()
    if version and version[0] >= 5:
        import google.protobuf
        raise RuntimeError(
            "TFLite conversion requires a protobuf version < 5 for this TensorFlow "
            f"stack. Found protobuf {google.protobuf.__version__}. "
            "Reinstall the environment with `protobuf>=3.20.3,<5` and retry."
        )


def _convert_saved_model_int8(
    model,
    representative_data_gen_fn,
    output_path: str | Path,
) -> bytes:
    """
    Convert via SavedModel export instead of `from_keras_model`.

    In this environment, the direct Keras converter path crashes in MLIR. The
    SavedModel route avoids that failure while still producing a fully INT8
    `.tflite` model suitable for Cortex-M deployment.
    """
    import tensorflow as tf

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="recho_denoiser_savedmodel_") as tmpdir:
        model.export(tmpdir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_data_gen_fn
        tflite_model = converter.convert()

    out.write_bytes(tflite_model)
    print(f"[convert_denoiser] TFLite INT8 model saved: {out} ({len(tflite_model):,} bytes)")
    return tflite_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the TCN denoiser to TFLite INT8.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/denoiser_tcn.keras")
    parser.add_argument("--firmware_dir", type=str, default="firmware/denoiser")
    parser.add_argument("--n_calib", type=int, default=24)
    args = parser.parse_args()

    _ensure_converter_environment()

    from pipeline.convert import (
        convert_to_tflite_int8,
        extract_cmsis_nn_params,
        generate_cmsis_nn_header,
        generate_model_data_cc,
        print_deployment_summary,
    )

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

    # Use an inference-only graph and export through SavedModel. This avoids a
    # direct Keras-to-TFLite MLIR failure observed with this TensorFlow stack.
    import tensorflow as tf

    export_model = tf.keras.Model(
        inputs=model._model.inputs,
        outputs=model._model.outputs,
        name="tcn_denoiser_export",
    )

    _convert_saved_model_int8(export_model, rep_gen, tflite_path)
    layer_params = extract_cmsis_nn_params(tflite_path)
    generate_cmsis_nn_header(layer_params, fw_dir / "cmsis_nn_params.h")
    generate_model_data_cc(tflite_path, fw_dir / "model_data.cc")
    print_deployment_summary(tflite_path)


if __name__ == "__main__":
    main()
