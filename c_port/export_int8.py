"""Export a trained Neuro-C Keras model to int8 quantized C headers.

Usage:
    python3 export_int8.py <trained_model.keras> <output_header.h> [--calib-size N]

What this does:
    1. Loads the float .keras Neuro-C model.
    2. Runs a calibration pass through it on N random training samples
       (default: 500) to record the activation range of each layer.
    3. Computes per-layer activation scales (symmetric int8).
    4. Quantizes per-neuron scale and bias for each layer as int16/int32
       with a per-layer shift exponent.
    5. Emits a C header consumable by neuroc_int8.c.

The math (see neuroc_int8.h for details):
    real_value ≈ activation_scale * int8_value
    M[j]       = (input_scale / output_scale) * w_j
    scale_q[j] = round(M[j] * 2^shift)         (int16)
    bias_q[j]  = round(b_j * 2^shift / output_scale)   (int32)

Output of one layer:
    out_int8[j] = saturate( ((scale_q[j] * acc + bias_q[j]) + 2^(shift-1)) >> shift )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Re-use the float exporter's helpers + NeuroCLayer definition.
from export_model import _emit_array, _activation_name, NeuroCLayer, ternary_quantize  # noqa: E402

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402


# -------------------------------------------------------------------------
# Calibration: capture each layer's float activation range on a sample set
# -------------------------------------------------------------------------
def collect_activation_ranges(model, x_calib):
    """For each NeuroCLayer in `model`, run x_calib through the network
    and record (min, max) of the layer's output activations.

    Approach: call each NeuroCLayer manually in sequence. This avoids
    relying on the Keras-3 Sequential model's `.input`/`.output` symbolic
    graph which doesn't always materialise for runtime-built layers.

    Returns:
        input_range:   (min, max) of the model's input
        output_ranges: list of (min, max) per NeuroCLayer
    """
    neuroc_layers = [l for l in model.layers if isinstance(l, NeuroCLayer)]

    x = tf.convert_to_tensor(x_calib, dtype=tf.float32)
    output_ranges = []
    for layer in neuroc_layers:
        x = layer(x)            # forward through this layer using its trained weights
        arr = x.numpy()
        output_ranges.append((float(arr.min()), float(arr.max())))

    input_range = (float(x_calib.min()), float(x_calib.max()))
    return input_range, output_ranges


def symmetric_scale(min_v: float, max_v: float, eps: float = 1e-8) -> float:
    """Return the float multiplier that maps real values to symmetric int8
    [-127, 127]. We use 127 (not 128) so the scale is symmetric and there
    is no asymmetric zero-point complexity."""
    m = max(abs(min_v), abs(max_v), eps)
    return m / 127.0


# -------------------------------------------------------------------------
# Per-layer quantization
# -------------------------------------------------------------------------
def quantize_layer(layer: NeuroCLayer,
                   input_scale: float,
                   output_scale: float,
                   target_scale_q_max: int = 16384):
    """Compute (scale_q[], bias_q[], shift) for one NeuroCLayer.

    Math:
        M[j]       = (input_scale / output_scale) * w_j         (float, signed)
        scale_q[j] = round(M[j] * 2^shift)                       (int16)
        bias_q[j]  = round(b_j * 2^shift / output_scale)         (int32)

    We pick the shift so that the LARGEST |scale_q[j]| is near
    `target_scale_q_max` (default ~half of int16 range, giving headroom).
    """
    w = layer.scale.numpy().astype(np.float64)                # (M,)
    b = layer.bias.numpy().astype(np.float64)                 # (M,)

    M = (input_scale / output_scale) * w                      # (M,)
    max_abs_M = max(np.max(np.abs(M)), 1e-12)

    # shift such that target_scale_q_max ≈ max_abs_M * 2^shift
    shift = int(np.round(np.log2(target_scale_q_max / max_abs_M)))
    # Clamp shift to a reasonable range; shift=0 means no shift, shift=30
    # is the practical max (int32 has 31 bits of magnitude).
    if shift < 0:
        shift = 0
    if shift > 30:
        shift = 30

    scale_factor = float(1 << shift)
    scale_q = np.round(M * scale_factor).astype(np.int64)
    # Clip to int16 range
    scale_q = np.clip(scale_q, -32768, 32767).astype(np.int16)

    # Bias must be in the SAME post-multiply units, so multiply by scale_factor
    # AND divide by output_scale (because final output is real / output_scale).
    bias_q = np.round((b / output_scale) * scale_factor).astype(np.int64)
    # Clip to int32 range
    INT32_MAX = (1 << 31) - 1
    INT32_MIN = -(1 << 31)
    bias_q = np.clip(bias_q, INT32_MIN, INT32_MAX).astype(np.int32)

    return scale_q, bias_q, shift


# -------------------------------------------------------------------------
# Main exporter
# -------------------------------------------------------------------------
def export_int8(model_path: str, header_path: str, calib_size: int = 500,
                dataset_override: str = "") -> None:
    print(f"Loading {model_path} ...")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "NeuroCLayer": NeuroCLayer,
            "ternary_quantize": ternary_quantize,
        },
        compile=False,
    )

    neuroc_layers = [l for l in model.layers if isinstance(l, NeuroCLayer)]
    if not neuroc_layers:
        raise RuntimeError("Model contains no NeuroCLayer instances.")

    input_size  = int(neuroc_layers[0].latent_kernel.shape[0])
    output_size = int(neuroc_layers[-1].units)
    max_width   = max(int(l.units) for l in neuroc_layers)

    # ---------- Calibration ----------
    print(f"Calibrating on {calib_size} samples ...")
    # Use --dataset flag if provided, otherwise infer from input_size.
    ds = (dataset_override or "").lower()
    if ds == "":
        if input_size == 784:
            ds = "mnist"
        elif input_size == 3072:
            ds = "cifar5"
        else:
            raise RuntimeError(
                f"Don't know which dataset has input_size={input_size}; "
                f"pass --dataset explicitly.")

    if ds == "mnist":
        (x_train, _), _ = keras.datasets.mnist.load_data()
        x_calib = x_train[:calib_size].astype("float32") / 255.0
        x_calib = x_calib.reshape(-1, input_size)
    elif ds == "fmnist":
        (x_train, _), _ = keras.datasets.fashion_mnist.load_data()
        x_calib = x_train[:calib_size].astype("float32") / 255.0
        x_calib = x_calib.reshape(-1, input_size)
    elif ds == "cifar5":
        (x_train, y_train), _ = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        mask = y_train < 5            # only the first 5 classes
        x_train = x_train[mask]
        mean = np.array([0.4914, 0.4822, 0.4465]) * 255
        std  = np.array([0.2470, 0.2435, 0.2616]) * 255
        x_calib = (x_train[:calib_size].astype("float32") - mean) / std
        x_calib = x_calib.reshape(-1, input_size)
    elif ds == "har":
        # Lazy import so MNIST runs don't pay HAR's download cost.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "har_comparison"))
        from har_data import load_har as _load_har   # noqa: E402
        x_train, _, _, _ = _load_har()
        x_calib = x_train[:calib_size].astype("float32")
    elif ds == "kws":
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kws_comparison"))
        from kws_data import load_kws as _load_kws   # noqa: E402
        x_train, _, _, _ = _load_kws()
        x_calib = x_train[:calib_size].astype("float32")
    else:
        raise RuntimeError(f"Unknown dataset {ds!r}")

    input_range, output_ranges = collect_activation_ranges(model, x_calib)
    print(f"  Input range  : [{input_range[0]:.4f}, {input_range[1]:.4f}]")
    for i, r in enumerate(output_ranges):
        print(f"  Layer {i} range: [{r[0]:.4f}, {r[1]:.4f}]")

    # ---------- Compute per-layer scales ----------
    input_scale = symmetric_scale(*input_range)
    output_scales = [symmetric_scale(*r) for r in output_ranges]
    print(f"\n  input_scale  = {input_scale:.6e}")
    for i, s in enumerate(output_scales):
        print(f"  output_scale[{i}] = {s:.6e}")

    # ---------- Per-layer quantization ----------
    # Layer i's input_scale is the model input scale (i=0) or the previous
    # layer's output_scale (i>0).
    layer_quants = []
    cur_in_scale = input_scale
    for i, layer in enumerate(neuroc_layers):
        cur_out_scale = output_scales[i]
        scale_q, bias_q, shift = quantize_layer(layer, cur_in_scale, cur_out_scale)
        layer_quants.append((scale_q, bias_q, shift))
        cur_in_scale = cur_out_scale   # next layer's input scale = this layer's output scale

    # ---------- Emit C header ----------
    lines: list[str] = []
    lines.append("/* AUTO-GENERATED by c_port/export_int8.py — do not edit. */")
    lines.append("")
    lines.append("#ifndef NEUROC_INT8_MODEL_DATA_H")
    lines.append("#define NEUROC_INT8_MODEL_DATA_H")
    lines.append("")
    lines.append('#include "neuroc_int8.h"')
    lines.append("")
    lines.append(f"#define MODEL_INPUT_SIZE       {input_size}")
    lines.append(f"#define MODEL_OUTPUT_SIZE      {output_size}")
    lines.append(f"#define MODEL_MAX_LAYER_WIDTH  {max_width}")
    lines.append("")
    # Emit the calibrated input scale so the deployment can convert float
    # inputs to int8 if needed.
    lines.append(f"#define MODEL_INPUT_SCALE      {input_scale!r}f")
    lines.append("")

    BLOCK_SIZE = 256
    layer_struct_inits: list[str] = []

    for i, layer in enumerate(neuroc_layers):
        kernel = ternary_quantize(layer.latent_kernel).numpy()
        in_dim, units = kernel.shape
        num_blocks = (in_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

        pos_col_ptr_all, neg_col_ptr_all = [], []
        pos_indices_all, neg_indices_all = [], []
        pos_block_offset, neg_block_offset = [0], [0]

        for b in range(num_blocks):
            block_start = b * BLOCK_SIZE
            block_end   = min(block_start + BLOCK_SIZE, in_dim)
            block       = kernel[block_start:block_end, :]

            block_pos_cp, block_neg_cp = [0], [0]
            block_pos_idx, block_neg_idx = [], []
            for j in range(units):
                col = block[:, j]
                p = np.flatnonzero(col > 0).tolist()
                n = np.flatnonzero(col < 0).tolist()
                block_pos_idx.extend(p)
                block_neg_idx.extend(n)
                block_pos_cp.append(len(block_pos_idx))
                block_neg_cp.append(len(block_neg_idx))

            pos_col_ptr_all.extend(block_pos_cp)
            neg_col_ptr_all.extend(block_neg_cp)
            pos_indices_all.extend(block_pos_idx)
            neg_indices_all.extend(block_neg_idx)
            pos_block_offset.append(len(pos_indices_all))
            neg_block_offset.append(len(neg_indices_all))

        scale_q, bias_q, shift = layer_quants[i]
        act_serialized = keras.activations.serialize(layer.activation)
        # The final layer's activation in the int8 engine is always NONE
        # (we want raw int32 logits for argmax). Hidden ReLU layers stay RELU.
        if i == len(neuroc_layers) - 1:
            int8_act_name = "NEUROC_INT8_ACT_NONE"
        elif act_serialized == "relu":
            int8_act_name = "NEUROC_INT8_ACT_RELU"
        else:
            int8_act_name = "NEUROC_INT8_ACT_NONE"

        n_pos = len(pos_indices_all)
        n_neg = len(neg_indices_all)
        n_total = in_dim * units
        n_zero = n_total - n_pos - n_neg
        print(f"  Layer {i:>2}  {in_dim:>5} -> {units:<5}  blocks={num_blocks}  "
              f"+1={n_pos:>6,}  -1={n_neg:>6,}  0={n_zero/n_total:.1%}  "
              f"shift={shift}  scale_q[{np.min(scale_q)}..{np.max(scale_q)}]")

        lines.append(f"/* ---- Layer {i}: {in_dim} -> {units}, {act_serialized} ---- */")
        lines.append(_emit_array(f"L{i}_pos_col_ptr",      "uint16_t", pos_col_ptr_all))
        lines.append(_emit_array(f"L{i}_pos_block_offset", "uint32_t", pos_block_offset))
        lines.append(_emit_array(f"L{i}_pos_indices",      "uint8_t",  pos_indices_all))
        lines.append(_emit_array(f"L{i}_neg_col_ptr",      "uint16_t", neg_col_ptr_all))
        lines.append(_emit_array(f"L{i}_neg_block_offset", "uint32_t", neg_block_offset))
        lines.append(_emit_array(f"L{i}_neg_indices",      "uint8_t",  neg_indices_all))
        lines.append(_emit_array(f"L{i}_scale_q", "int16_t", scale_q.tolist()))
        lines.append(_emit_array(f"L{i}_bias_q",  "int32_t", bias_q.tolist()))
        lines.append("")

        layer_struct_inits.append(
            "    {\n"
            f"        .input_size       = {in_dim},\n"
            f"        .output_size      = {units},\n"
            f"        .block_size       = {BLOCK_SIZE},\n"
            f"        .num_blocks       = {num_blocks},\n"
            f"        .pos_col_ptr      = L{i}_pos_col_ptr,\n"
            f"        .pos_block_offset = L{i}_pos_block_offset,\n"
            f"        .pos_indices      = L{i}_pos_indices,\n"
            f"        .neg_col_ptr      = L{i}_neg_col_ptr,\n"
            f"        .neg_block_offset = L{i}_neg_block_offset,\n"
            f"        .neg_indices      = L{i}_neg_indices,\n"
            f"        .scale_q          = L{i}_scale_q,\n"
            f"        .bias_q           = L{i}_bias_q,\n"
            f"        .shift            = {shift},\n"
            f"        .activation       = {int8_act_name},\n"
            "    }"
        )

    lines.append("static const neuroc_int8_layer_t MODEL_LAYERS[] = {")
    lines.append(",\n".join(layer_struct_inits))
    lines.append("};")
    lines.append("")
    lines.append("static const neuroc_int8_model_t MODEL = {")
    lines.append(f"    .num_layers      = {len(neuroc_layers)},")
    lines.append(f"    .max_layer_width = {max_width},")
    lines.append("    .layers          = MODEL_LAYERS,")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* NEUROC_INT8_MODEL_DATA_H */")

    out_path = Path(header_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"\nWrote {out_path}  ({size_kb:.1f} KB of C source)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="Path to a trained .keras model file")
    ap.add_argument("header_path", help="Where to write the C header")
    ap.add_argument("--calib-size", type=int, default=500,
                    help="How many training samples to use for activation-range "
                         "calibration (default: 500)")
    ap.add_argument("--dataset", default="",
                    help="Dataset key for calibration: mnist | fmnist | cifar5 | har | kws. "
                         "Auto-detected from input dimension if omitted (mnist for 784, "
                         "cifar5 for 3072).")
    args = ap.parse_args()
    export_int8(args.model_path, args.header_path, args.calib_size, args.dataset)
