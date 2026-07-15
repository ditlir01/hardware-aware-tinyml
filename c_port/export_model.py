"""Export a trained Neuro-C Keras model to a C header consumable by neuroc.c.

Usage:
    python3 export_model.py <trained_model.keras> <output_header.h>

The script reproduces the same TWN-style threshold quantization the training
loop uses (threshold = 0.5 * mean(|w|)), then for every NeuroCLayer emits the
CSC index arrays for the +1 and -1 connections separately, plus the float
scale and bias vectors. The result is a single .h file that compiles
alongside neuroc.c with no other dependencies.

Layout of the emitted header (excerpt):

    static const uint32_t L0_pos_col_ptr[] = { ... };
    static const uint16_t L0_pos_indices[] = { ... };
    static const uint32_t L0_neg_col_ptr[] = { ... };
    static const uint16_t L0_neg_indices[] = { ... };
    static const float    L0_scale[]       = { ... };
    static const float    L0_bias[]        = { ... };

    static const neuroc_layer_t MODEL_LAYERS[] = { ..., ... };
    static const neuroc_model_t MODEL = { ... };

The exporter also writes a small `MODEL_INPUT_SIZE` / `MODEL_OUTPUT_SIZE` /
`MODEL_MAX_LAYER_WIDTH` set of #defines so the C main can size its buffers
without inspecting the model struct at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# Make sure we can import NeuroCLayer from any of the paper-faithful scripts.
HERE = Path(__file__).resolve().parent
KERAS_EXAMPLES = HERE.parent / "keras_examples"
sys.path.insert(0, str(KERAS_EXAMPLES))

# Import keras only once we've set up sys.path so we can resolve the
# NeuroCLayer custom object below.
try:
    import tensorflow as tf  # noqa: E402
    from tensorflow import keras  # noqa: E402
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required to run export_model.py"
    ) from exc


# -------------------------------------------------------------------------
# Re-define the NeuroCLayer + ternary_quantize here so we don't depend on a
# specific script's filename. (Keras needs them to be in scope when loading
# a .keras archive that uses them as custom_objects.)
# -------------------------------------------------------------------------
def ternary_quantize(latent):
    threshold = 0.5 * tf.reduce_mean(tf.abs(latent))
    q = tf.zeros_like(latent)
    q = tf.where(latent >  threshold,  tf.ones_like(latent), q)
    q = tf.where(latent < -threshold, -tf.ones_like(latent), q)
    latent_clipped = tf.clip_by_value(latent, -1.0, 1.0)
    return tf.stop_gradient(q - latent_clipped) + latent_clipped


class NeuroCLayer(keras.layers.Layer):
    """NeuroC FC layer — must match the definition in the *_neuroc_paper.py
    files in this repo. Two scale_init modes:
      - None (default): scale = Constant(1/sqrt(fan_in)).  This is what the
        current MNIST / FashionMNIST / CIFAR-5 / HAR / KWS files use.
      - (low, high) tuple: scale = RandomUniform(low, high).  Used by older
        mnist_neuroc.py / cifar5_neuroc.py files.
    The exporter has to recognise both because Keras deserializes whatever
    is in the saved model file's config — if you trained a model with
    scale_init=None and the exporter only understood tuples, deserialization
    would crash before we ever read the trained weights.
    """
    def __init__(self, units, activation=None, scale_init=None, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.activation = keras.activations.get(activation)
        self.scale_init = scale_init   # None or (low, high) tuple

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.latent_kernel = self.add_weight(
            name="latent_kernel", shape=(in_dim, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        if self.scale_init is None:
            scale_initializer = keras.initializers.Constant(1.0 / np.sqrt(in_dim))
        else:
            scale_initializer = keras.initializers.RandomUniform(*self.scale_init)
        self.scale = self.add_weight(
            name="scale", shape=(self.units,),
            initializer=scale_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,),
            initializer="zeros", trainable=True,
        )

    def call(self, x):
        A = ternary_quantize(self.latent_kernel)
        z = tf.matmul(x, A)
        z = self.scale * z + self.bias
        return self.activation(z)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "scale_init": self.scale_init,   # serializes as null or [low, high]
        })
        return cfg


# -------------------------------------------------------------------------
# Helpers for emitting C arrays
# -------------------------------------------------------------------------
def _emit_array(name: str, ctype: str, values, per_line: int = 16) -> str:
    """Format `values` as a C array literal of type `ctype`."""
    lines = [f"static const {ctype} {name}[{len(values)}] = {{"]
    fmt = "%a" if ctype.startswith("float") else "%d"
    chunk = []
    for i, v in enumerate(values):
        chunk.append(fmt % v)
        if (i + 1) % per_line == 0:
            lines.append("    " + ", ".join(chunk) + ",")
            chunk = []
    if chunk:
        lines.append("    " + ", ".join(chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def _activation_name(act_serialized: str) -> str:
    """Map a Keras activation name to a NEUROC_ACT_* enum value."""
    if act_serialized in (None, "linear"):
        return "NEUROC_ACT_NONE"
    if act_serialized == "relu":
        return "NEUROC_ACT_RELU"
    if act_serialized == "softmax":
        return "NEUROC_ACT_SOFTMAX"
    raise ValueError(f"Unsupported activation for C port: {act_serialized!r}")


# -------------------------------------------------------------------------
# Main exporter
# -------------------------------------------------------------------------
def export_model(model_path: str, header_path: str) -> None:
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

    # Sanity: every NeuroCLayer should map to a single C layer.
    max_width = max(int(l.units) for l in neuroc_layers)
    input_size = int(neuroc_layers[0].latent_kernel.shape[0])
    output_size = int(neuroc_layers[-1].units)

    lines: list[str] = []
    lines.append("/* AUTO-GENERATED by c_port/export_model.py — do not edit. */")
    lines.append("")
    lines.append("#ifndef NEUROC_MODEL_DATA_H")
    lines.append("#define NEUROC_MODEL_DATA_H")
    lines.append("")
    lines.append('#include "neuroc.h"')
    lines.append("")
    lines.append(f"#define MODEL_INPUT_SIZE       {input_size}")
    lines.append(f"#define MODEL_OUTPUT_SIZE      {output_size}")
    lines.append(f"#define MODEL_MAX_LAYER_WIDTH  {max_width}")
    lines.append("")

    layer_struct_inits: list[str] = []

    BLOCK_SIZE = 256   # paper Sec 4.2 — keeps indices in uint8_t range

    for i, layer in enumerate(neuroc_layers):
        # Ternarize the latent kernel exactly as the training-time forward pass.
        kernel = ternary_quantize(layer.latent_kernel).numpy()  # shape (in_dim, units)
        scale  = layer.scale.numpy().astype(np.float32)
        bias   = layer.bias.numpy().astype(np.float32)
        in_dim, units = kernel.shape

        # Partition the input into BLOCK_SIZE-sized blocks. Each block's
        # connection indices fit in uint8_t (0..BLOCK_SIZE-1).
        num_blocks = (in_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

        if BLOCK_SIZE > 256:
            raise ValueError("BLOCK_SIZE must be <= 256 to keep indices in uint8 range.")

        # For each block, build its own col_ptr table (length units+1) and
        # its own concatenated index array. Then concatenate all block
        # col_ptrs and all block indices, and remember per-block offsets.
        pos_col_ptr_all: list[int] = []
        neg_col_ptr_all: list[int] = []
        pos_indices_all: list[int] = []
        neg_indices_all: list[int] = []
        pos_block_offset: list[int] = [0]
        neg_block_offset: list[int] = [0]

        for b in range(num_blocks):
            block_start = b * BLOCK_SIZE
            block_end   = min(block_start + BLOCK_SIZE, in_dim)
            block       = kernel[block_start:block_end, :]   # (block_size, units)

            block_pos_cp = [0]
            block_neg_cp = [0]
            block_pos_idx: list[int] = []
            block_neg_idx: list[int] = []
            for j in range(units):
                col = block[:, j]
                p = np.flatnonzero(col > 0).tolist()   # block-local indices in [0, block_size)
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

        # Stats — useful for sanity-checking against the Keras run.
        n_pos = len(pos_indices_all)
        n_neg = len(neg_indices_all)
        n_total = in_dim * units
        n_zero = n_total - n_pos - n_neg
        act_serialized = keras.activations.serialize(layer.activation)
        print(f"  Layer {i:>2}  {in_dim:>5} -> {units:<5}  "
              f"blocks={num_blocks}  +1={n_pos:>7,}  -1={n_neg:>7,}  "
              f"0={n_zero:>7,} ({n_zero/n_total:.1%})  act={act_serialized}")

        # Cumulative pointer values can exceed uint16 only across many blocks
        # — within one block they are bounded by block_size*units which we
        # keep under 2^16 by using num_blocks separate col_ptr arrays. Each
        # block's col_ptr table values therefore fit in uint16.
        max_pos_cp_in_block = max(
            pos_block_offset[b+1] - pos_block_offset[b] for b in range(num_blocks)
        )
        max_neg_cp_in_block = max(
            neg_block_offset[b+1] - neg_block_offset[b] for b in range(num_blocks)
        )
        if max_pos_cp_in_block > 65535 or max_neg_cp_in_block > 65535:
            raise ValueError(f"Layer {i}: per-block non-zero count exceeds uint16 range.")

        # Emit arrays.
        lines.append(f"/* ---- Layer {i}: {in_dim} -> {units}, "
                     f"{num_blocks} blocks x {BLOCK_SIZE}, {act_serialized} ---- */")
        lines.append(_emit_array(f"L{i}_pos_col_ptr",      "uint16_t", pos_col_ptr_all))
        lines.append(_emit_array(f"L{i}_pos_block_offset", "uint32_t", pos_block_offset))
        lines.append(_emit_array(f"L{i}_pos_indices",      "uint8_t",  pos_indices_all))
        lines.append(_emit_array(f"L{i}_neg_col_ptr",      "uint16_t", neg_col_ptr_all))
        lines.append(_emit_array(f"L{i}_neg_block_offset", "uint32_t", neg_block_offset))
        lines.append(_emit_array(f"L{i}_neg_indices",      "uint8_t",  neg_indices_all))
        lines.append(_emit_array(f"L{i}_scale", "float", scale))
        lines.append(_emit_array(f"L{i}_bias",  "float", bias))
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
            f"        .scale            = L{i}_scale,\n"
            f"        .bias             = L{i}_bias,\n"
            f"        .activation       = {_activation_name(act_serialized)},\n"
            "    }"
        )

    lines.append("static const neuroc_layer_t MODEL_LAYERS[] = {")
    lines.append(",\n".join(layer_struct_inits))
    lines.append("};")
    lines.append("")
    lines.append("static const neuroc_model_t MODEL = {")
    lines.append(f"    .num_layers      = {len(neuroc_layers)},")
    lines.append(f"    .max_layer_width = {max_width},")
    lines.append("    .layers          = MODEL_LAYERS,")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* NEUROC_MODEL_DATA_H */")

    out_path = Path(header_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    size_kb = out_path.stat().st_size / 1024.0
    print(f"Wrote {out_path}  ({size_kb:.1f} KB of C source)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="Path to a trained .keras model file")
    ap.add_argument("header_path", help="Where to write the C header (e.g. model_data.h)")
    args = ap.parse_args()
    export_model(args.model_path, args.header_path)
