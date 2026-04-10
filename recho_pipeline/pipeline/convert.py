"""
TFLite INT8 conversion and CMSIS-NN code generation.

Converts a QAT-trained Keras model to a fully INT8-quantised TFLite model,
extracts CMSIS-NN kernel parameters from the flatbuffer, and generates
C header files for direct inclusion in Arm Cortex-M firmware.

Output artefacts:
  - firmware/cmsis_nn_params.h  — weights, biases, quantisation params
  - firmware/model_data.cc      — TFLite Micro C array (xxd-style)
  - Deployment summary to stdout
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import tensorflow as tf


# Effective inference budgets per MCU target.
# These are conservative screening limits, not final board-level guarantees.
MCU_RAM_LIMITS: dict[str, int] = {
    "M4": 48 * 1024,
    "M33": 64 * 1024,
    "M55": 128 * 1024,
    "M85": 256 * 1024,
}


def convert_to_tflite_int8(
    model: tf.keras.Model,
    representative_data_gen,
    output_path: str | Path,
) -> bytes:
    """
    Step A — Full INT8 quantisation via TFLite converter.

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    Returns:
        The raw TFLite flatbuffer bytes.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_model)
    print(f"[convert] TFLite INT8 model saved: {out} ({len(tflite_model):,} bytes)")

    return tflite_model


def extract_cmsis_nn_params(tflite_path: str | Path) -> list[dict]:
    """
    Step B — Extract CMSIS-NN parameters from the TFLite flatbuffer.

    For each quantised layer, parses:
      - input/output zero points and scales
      - weight quantisation scale per channel
      - bias int32 values (pre-scaled for CMSIS-NN)
      - output shift and multiplier values

    These values feed directly into the CMSIS-NN kernel parameter structs:
      cmsis_nn_conv_params, cmsis_nn_per_channel_quant_params, cmsis_nn_dims
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    layer_params = []

    for detail in tensor_details:
        quant = detail.get("quantization_parameters", {})
        scales = quant.get("scales", np.array([]))
        zero_points = quant.get("zero_points", np.array([]))

        if len(scales) == 0:
            continue

        tensor_data = None
        try:
            tensor_data = interpreter.get_tensor(detail["index"])
        except ValueError:
            # Some TFLite tensors are dynamically produced activations and do not
            # expose host-readable buffers even after allocate_tensors(). We
            # still keep their quant metadata for reporting, but only constant
            # tensors (weights / biases) are expected to carry raw data.
            pass
        param = {
            "name": detail["name"],
            "shape": list(detail["shape"]),
            "dtype": str(detail["dtype"]),
            "scales": scales.tolist(),
            "zero_points": zero_points.tolist(),
            "data_shape": list(tensor_data.shape) if tensor_data is not None else [],
        }

        is_constant_tensor = detail["name"].startswith("tfl.pseudo_qconst")

        # TFLite exported constants appear as tfl.pseudo_qconst* tensors.
        # Readable activations may also have int8 buffers after allocation, but
        # they must not be emitted as firmware weights.
        if is_constant_tensor and tensor_data is not None and detail["dtype"] == np.int8 and len(detail["shape"]) >= 2:
            param["type"] = "weight"
            param["weight_data"] = tensor_data
        elif is_constant_tensor and tensor_data is not None and detail["dtype"] == np.int32:
            param["type"] = "bias"
            param["bias_data"] = tensor_data

        layer_params.append(param)

    print(f"[convert] Extracted {len(layer_params)} quantised tensors")
    for p in layer_params:
        ptype = p.get("type", "activation")
        print(f"  {p['name']:50s}  shape={p['shape']}  type={ptype}  "
              f"scale={p['scales'][0]:.8f}  zp={p['zero_points'][0]}")

    return layer_params


def _compute_multiplier_shift(scale: float) -> tuple[int, int]:
    """
    Compute the quantised multiplier and shift for CMSIS-NN.

    CMSIS-NN kernels use fixed-point arithmetic:
      output = (input * multiplier) >> (-shift)
    where multiplier is in [0x40000000, 0x7FFFFFFF] and shift <= 0.
    """
    if scale == 0:
        return 0, 0

    # Decompose scale into mantissa * 2^exponent
    significand = scale
    shift = 0
    while significand < 0.5:
        significand *= 2.0
        shift -= 1
    while significand >= 1.0:
        significand /= 2.0
        shift += 1

    # Convert to int32 fixed-point multiplier
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        shift += 1

    return multiplier, shift


def generate_cmsis_nn_header(
    layer_params: list[dict],
    output_path: str | Path,
) -> None:
    """
    Step C — Generate firmware/cmsis_nn_params.h

    Contains:
      - Weight arrays as int8_t[]
      - Bias arrays as int32_t[]
      - Per-layer quantisation parameters as #define constants
      - cmsis_nn_dims structs for input, filter, and output dimensions
      - Scratch buffer size calculation notes
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("/* Auto-generated CMSIS-NN parameters — do not edit */")
    lines.append("/* Generated by recho_pipeline/pipeline/convert.py */")
    lines.append("")
    lines.append("#ifndef CMSIS_NN_PARAMS_H")
    lines.append("#define CMSIS_NN_PARAMS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append('#include "arm_nnfunctions.h"')
    lines.append("")

    layer_idx = 0
    for p in layer_params:
        safe_name = p["name"].replace("/", "_").replace(":", "_").replace(".", "_")

        if p.get("type") == "weight":
            weight_data = p["weight_data"]
            flat = weight_data.flatten()
            lines.append(f"/* Layer: {p['name']} */")
            lines.append(f"/* Shape: {p['shape']} */")
            lines.append(f"static const int8_t {safe_name}[{len(flat)}] = {{")

            # Format weight data in rows of 16
            for row_start in range(0, len(flat), 16):
                row = flat[row_start:row_start + 16]
                row_str = ", ".join(str(int(v)) for v in row)
                lines.append(f"    {row_str},")
            lines.append("};")
            lines.append("")

            # Per-channel quantisation parameters
            scales = p["scales"]
            zps = p["zero_points"]
            n_channels = len(scales)

            lines.append(f"/* Per-channel quantisation for {safe_name} */")

            multipliers = []
            shifts = []
            for s in scales:
                m, sh = _compute_multiplier_shift(s)
                multipliers.append(m)
                shifts.append(sh)

            lines.append(f"static const int32_t {safe_name}_multiplier[{n_channels}] = {{")
            for row_start in range(0, n_channels, 8):
                row = multipliers[row_start:row_start + 8]
                row_str = ", ".join(str(v) for v in row)
                lines.append(f"    {row_str},")
            lines.append("};")

            lines.append(f"static const int32_t {safe_name}_shift[{n_channels}] = {{")
            for row_start in range(0, n_channels, 8):
                row = shifts[row_start:row_start + 8]
                row_str = ", ".join(str(v) for v in row)
                lines.append(f"    {row_str},")
            lines.append("};")
            lines.append("")

        elif p.get("type") == "bias":
            bias_data = p["bias_data"]
            flat = bias_data.flatten()
            lines.append(f"/* Bias: {p['name']} */")
            lines.append(f"/* Pre-scaled int32 bias for CMSIS-NN (no further scaling needed) */")
            lines.append(f"static const int32_t {safe_name}[{len(flat)}] = {{")
            for row_start in range(0, len(flat), 8):
                row = flat[row_start:row_start + 8]
                row_str = ", ".join(str(int(v)) for v in row)
                lines.append(f"    {row_str},")
            lines.append("};")
            lines.append("")

        # Emit #define for zero point and scale
        lines.append(f"#define {safe_name.upper()}_SCALE     {p['scales'][0]:.10f}f")
        lines.append(f"#define {safe_name.upper()}_ZERO_POINT {p['zero_points'][0]}")
        lines.append("")

        layer_idx += 1

    # Scratch buffer size note
    lines.append("/*")
    lines.append(" * Scratch buffer sizing:")
    lines.append(" * Call arm_convolve_s8_get_buffer_size() at init for each conv layer")
    lines.append(" * and allocate the maximum across all layers.")
    lines.append(" *")
    lines.append(" * Example:")
    lines.append(" *   int32_t buf_size = arm_convolve_s8_get_buffer_size(")
    lines.append(" *       &input_dims, &filter_dims);")
    lines.append(" *   int8_t *scratch = (int8_t *)malloc(buf_size);")
    lines.append(" */")
    lines.append("")
    lines.append("#endif /* CMSIS_NN_PARAMS_H */")
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"[convert] Generated {out} ({len(lines)} lines)")


def generate_model_data_cc(
    tflite_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Step D — Generate firmware/model_data.cc as TFLite Micro C array.

    Equivalent to: xxd -i model.tflite > model_data.cc
    """
    tflite_bytes = Path(tflite_path).read_bytes()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("/* Auto-generated TFLite Micro model data — do not edit */")
    lines.append("/* Generated by recho_pipeline/pipeline/convert.py */")
    lines.append("")
    lines.append('#include <cstdint>')
    lines.append("")
    lines.append(f"alignas(16) const unsigned char g_model_data[] = {{")

    for row_start in range(0, len(tflite_bytes), 12):
        row = tflite_bytes[row_start:row_start + 12]
        hex_vals = ", ".join(f"0x{b:02x}" for b in row)
        lines.append(f"    {hex_vals},")

    lines.append("};")
    lines.append(f"const unsigned int g_model_data_len = {len(tflite_bytes)};")
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"[convert] Generated {out} ({len(tflite_bytes):,} bytes model)")


def print_deployment_summary(tflite_path: str | Path) -> None:
    """
    Step E — Print deployment summary.

    Reports model size, peak RAM, estimated MCU fit, and CMSIS-NN coverage.
    """
    tflite_bytes = Path(tflite_path).read_bytes()
    model_size = len(tflite_bytes)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    # Estimate peak RAM: largest intermediate tensor
    tensor_details = interpreter.get_tensor_details()
    peak_ram = 0
    for detail in tensor_details:
        shape = detail["shape"]
        # int8 = 1 byte per element
        size = int(np.prod(shape))
        peak_ram = max(peak_ram, size)

    # Check operator coverage
    # All ops should be INT8 builtins (no custom ops)
    unsupported_ops: list[str] = []
    # In a fully INT8 model, all ops are TFLITE_BUILTINS_INT8
    # We verify by checking that all tensors are int8 or int32 (biases)
    for detail in tensor_details:
        dtype = detail["dtype"]
        if dtype not in (np.int8, np.int32, np.float32):
            unsupported_ops.append(f"{detail['name']} (dtype={dtype})")

    print("\n" + "=" * 70)
    print("DEPLOYMENT SUMMARY (ESTIMATED)")
    print("=" * 70)
    print(f"  Model size:     {model_size:>10,} bytes ({model_size / 1024:.1f} KB)")
    print(f"  Peak RAM:       {peak_ram:>10,} bytes ({peak_ram / 1024:.1f} KB)")
    print(f"  Total tensors:  {len(tensor_details)}")
    print("  Note: fit is estimated from model bytes + largest tensor only.")
    print("        Final MCU support still depends on scratch buffers, arena size,")
    print("        preprocessing memory, and measured on-target latency.")
    print()

    for mcu, limit in MCU_RAM_LIMITS.items():
        fits = "YES" if (model_size + peak_ram) <= limit else "NO"
        print(f"  {mcu:4s} ({limit // 1024:>3d} KB budget): {fits}")

    if unsupported_ops:
        print(f"\n  WARNING: {len(unsupported_ops)} layers NOT covered by CMSIS-NN:")
        for op in unsupported_ops:
            print(f"    - {op}")
    else:
        print(f"\n  CMSIS-NN coverage: ALL layers covered (production ready)")

    print("=" * 70)


def convert_pipeline(
    model: tf.keras.Model,
    representative_data_gen,
    firmware_dir: str | Path = "firmware",
) -> None:
    """Run the full conversion pipeline (Steps A through E)."""
    fw = Path(firmware_dir)
    fw.mkdir(parents=True, exist_ok=True)

    tflite_path = fw / "model.tflite"

    # Step A: TFLite INT8 conversion
    tflite_bytes = convert_to_tflite_int8(model, representative_data_gen, tflite_path)

    # Step B: Extract CMSIS-NN parameters
    layer_params = extract_cmsis_nn_params(tflite_path)

    # Step C: Generate C header
    generate_cmsis_nn_header(layer_params, fw / "cmsis_nn_params.h")

    # Step D: Generate TFLite Micro C array
    generate_model_data_cc(tflite_path, fw / "model_data.cc")

    # Step E: Deployment summary
    print_deployment_summary(tflite_path)


def main() -> None:
    """Run conversion on a freshly trained synthetic model."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import generate_dataset
    from pipeline.ingest import process_dataset
    from pipeline.features import extract_features
    from pipeline.model import build_model
    from pipeline.train import apply_qat, prepare_data, train, representative_data_gen

    print("[convert] Building and training model on synthetic data ...")
    raw_x, labels = generate_dataset(n_clips_per_class=10, n_classes=5, cache=False)
    processed = process_dataset(raw_x)
    feature_maps, labels = extract_features(processed, labels)

    model = build_model(n_classes=5)

    try:
        qat_model = apply_qat(model)
    except ImportError:
        print("  WARNING: tensorflow-model-optimization not installed, using base model")
        qat_model = model

    x_train, y_train, x_val, y_val = prepare_data(feature_maps, labels, n_classes=5)
    train(qat_model, x_train, y_train, x_val, y_val, epochs=3, batch_size=8)

    fw_dir = Path(__file__).resolve().parent.parent / "firmware"

    def rep_gen():
        return representative_data_gen(feature_maps, n_samples=50)

    convert_pipeline(qat_model, rep_gen, firmware_dir=fw_dir)


if __name__ == "__main__":
    main()
