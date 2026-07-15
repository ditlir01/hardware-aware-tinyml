"""Verify that int8 quantization didn't crater accuracy.

Simulates the C int8 forward pass byte-for-byte in numpy, then evaluates
on the full test set and compares against the float model's accuracy.

Usage:
    python3 verify_int8.py <trained_model.keras>

Output:
    Float model test accuracy: XX.XX%
    Int8  model test accuracy: XX.XX%
    Disagreement rate (preds differ): XX.XX%
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_model import NeuroCLayer, ternary_quantize  # noqa: E402
from export_int8 import collect_activation_ranges, symmetric_scale, quantize_layer  # noqa: E402

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402


def simulate_int8_forward(layer_quants, ternary_kernels, biases_unused, scales_unused,
                          input_int8: np.ndarray) -> np.ndarray:
    """Run the int8 forward pass in numpy, byte-for-byte matching neuroc_int8.c.

    Args:
        layer_quants: list of (scale_q[M], bias_q[M], shift) per layer
        ternary_kernels: list of (in_dim, units) int8 matrices in {-1, 0, +1}
        input_int8: shape (N, input_dim), int8 inputs

    Returns:
        output_logits: shape (N, num_classes), int32 raw logits from final layer
    """
    N_layers = len(layer_quants)
    x = input_int8.astype(np.int32)   # promote to int32 for the math

    for layer_idx in range(N_layers):
        is_final = (layer_idx == N_layers - 1)
        scale_q, bias_q, shift = layer_quants[layer_idx]
        kernel = ternary_kernels[layer_idx]              # (in_dim, M), {-1, 0, +1}
        in_dim, M = kernel.shape

        # acc[N, M] = x @ kernel  (where + contributes positively, - negatively)
        acc = x @ kernel.astype(np.int32)                # (N, M)

        # Apply per-neuron scale + bias (broadcasting over N)
        scaled = acc * scale_q.astype(np.int32) + bias_q.astype(np.int32)

        # Rounded right shift
        round_term = (1 << (shift - 1)) if shift > 0 else 0
        shifted = (scaled + round_term) >> shift         # int32 arithmetic shift

        if is_final:
            # Final layer leaves raw logits
            x = shifted.astype(np.int32)
        else:
            # ReLU + int8 saturate
            shifted = np.where(shifted < 0, 0, shifted)
            shifted = np.clip(shifted, -128, 127)
            x = shifted.astype(np.int32)   # keep as int32 for next layer's matmul

    return x   # final output_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--calib-size", type=int, default=500)
    ap.add_argument("--dataset", default="",
                    help="mnist | fmnist | cifar5 | har | kws (auto-detected if omitted)")
    args = ap.parse_args()

    print(f"Loading {args.model_path} ...")
    model = keras.models.load_model(
        args.model_path,
        custom_objects={
            "NeuroCLayer": NeuroCLayer,
            "ternary_quantize": ternary_quantize,
        },
        compile=False,
    )

    neuroc_layers = [l for l in model.layers if isinstance(l, NeuroCLayer)]
    input_size = int(neuroc_layers[0].latent_kernel.shape[0])

    ds = (args.dataset or "").lower()
    if ds == "":
        if input_size == 784:
            ds = "mnist"
        elif input_size == 3072:
            ds = "cifar5"
        else:
            raise RuntimeError(f"Pass --dataset (input_size={input_size})")

    # Load training + test data for the chosen dataset.
    if ds == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test  = x_test.astype("float32")  / 255
        x_train = x_train.reshape(-1, input_size); x_test = x_test.reshape(-1, input_size)
    elif ds == "fmnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test  = x_test.astype("float32")  / 255
        x_train = x_train.reshape(-1, input_size); x_test = x_test.reshape(-1, input_size)
    elif ds == "cifar5":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten(); y_test = y_test.flatten()
        mask_train = y_train < 5; mask_test = y_test < 5
        x_train = x_train[mask_train]; y_train = y_train[mask_train]
        x_test  = x_test[mask_test];  y_test  = y_test[mask_test]
        mean = np.array([0.4914, 0.4822, 0.4465]) * 255
        std  = np.array([0.2470, 0.2435, 0.2616]) * 255
        x_train = (x_train.astype("float32") - mean) / std
        x_test  = (x_test.astype("float32") - mean) / std
        x_train = x_train.reshape(-1, input_size); x_test = x_test.reshape(-1, input_size)
    elif ds == "har":
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "har_comparison"))
        from har_data import load_har as _load_har   # noqa: E402
        x_train, y_train, x_test, y_test = _load_har()
        x_train = x_train.astype("float32"); x_test = x_test.astype("float32")
    elif ds == "kws":
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kws_comparison"))
        from kws_data import load_kws as _load_kws   # noqa: E402
        x_train, y_train, x_test, y_test = _load_kws()
        x_train = x_train.astype("float32"); x_test = x_test.astype("float32")
    else:
        raise RuntimeError(f"Unknown dataset {ds!r}")

    x_calib = x_train[:args.calib_size]

    # ---------- Float accuracy ----------
    print("Evaluating FLOAT model on test set ...")
    float_preds = model.predict(x_test, verbose=0).argmax(axis=1)
    float_acc = float(np.mean(float_preds == y_test))
    print(f"  Float test accuracy: {float_acc*100:.2f}%")

    # ---------- Quantize ----------
    print("Calibrating + quantizing ...")
    input_range, output_ranges = collect_activation_ranges(model, x_calib)
    input_scale = symmetric_scale(*input_range)
    output_scales = [symmetric_scale(*r) for r in output_ranges]

    layer_quants = []
    ternary_kernels = []
    cur_in_scale = input_scale
    for i, layer in enumerate(neuroc_layers):
        out_scale = output_scales[i]
        sq, bq, sh = quantize_layer(layer, cur_in_scale, out_scale)
        layer_quants.append((sq, bq, sh))
        ternary_kernels.append(ternary_quantize(layer.latent_kernel).numpy().astype(np.int8))
        cur_in_scale = out_scale

    # ---------- Int8 accuracy ----------
    print("Evaluating INT8 model on test set ...")
    # Quantize test inputs
    x_test_int8 = np.round(x_test / input_scale).astype(np.int32)
    x_test_int8 = np.clip(x_test_int8, -128, 127).astype(np.int8)
    int8_logits = simulate_int8_forward(layer_quants, ternary_kernels, None, None, x_test_int8)
    int8_preds = int8_logits.argmax(axis=1)
    int8_acc = float(np.mean(int8_preds == y_test))
    print(f"  Int8  test accuracy: {int8_acc*100:.2f}%")

    # ---------- Disagreement ----------
    disagree = float(np.mean(float_preds != int8_preds))
    print(f"\nDisagreement (float pred != int8 pred): {disagree*100:.2f}%")
    print(f"Accuracy delta              : {(int8_acc - float_acc)*100:+.2f} pp")


if __name__ == "__main__":
    main()
