# Neuro-C C port

Plain-C inference engine for the Neuro-C ternary-FC models trained by the
scripts in `../keras_examples/`. Builds on the host for verification and
cross-compiles for ARM Cortex-M0 (the paper's STM32F072RB target) with no
source-code changes.

## What's here

| File              | Role                                                            |
|-------------------|-----------------------------------------------------------------|
| `neuroc.h`        | Public API: `neuroc_model_t`, `neuroc_layer_t`, `neuroc_forward` |
| `neuroc.c`        | Inference engine (no multiplies in the inner loop)              |
| `export_model.py` | Convert a trained `*.keras` model → `model_data.h`              |
| `export_sample.py`| Dump one test-set input → `test_sample.h` for verification      |
| `example_main.c`  | Tiny demo: runs inference on one sample, prints predicted class |
| `Makefile`        | Host build                                                       |

## Quick start (host)

```bash
# 1. Train one of the paper-faithful models and save it
cd ../mnist_comparison
python3 mnist_neuroc_paper.py     # trains, prints accuracy
# (edit the script to add `model.save("mnist_neuroc.keras")` at the end)

# 2. Export weights + one test sample to C headers
cd ../c_port
python3 export_model.py  ../mnist_comparison/mnist_neuroc.keras  model_data.h
python3 export_sample.py mnist test_sample.h --index 0

# 3. Compile and run
make run
```

You should see the predicted class match the true label (and match what
Keras predicts on the same sample).

### Datasets supported by `export_sample.py`

| Dataset key | Input dim | Classes | Preprocessing source |
|---|---|---|---|
| `mnist` | 784 | 10 | inline in this script |
| `fmnist` | 784 | 10 | inline in this script |
| `cifar5` | 3072 | 5 | inline in this script (per-channel std) |
| `har` | 561 | 6 | reuses `har_comparison/har_data.py` |
| `kws` | 490 | 10 | reuses `kws_comparison/kws_data.py` |

Using the shared data modules for HAR and KWS guarantees no preprocessing
drift between Python training and C inference verification. The C engine
sees the exact same input vector the training pipeline produced.

## What this port covers

- **Ternary adjacency matrix**, exactly as the paper's Eq. 1: each output
  neuron has its own list of `+1` connections and its own list of `-1`
  connections. The forward pass accumulates inputs over those index sets
  with **no multiplications**.
- **Per-neuron scale `w_j`**, applied once per output neuron after the
  ternary accumulation — the only multiply in the whole forward pass.
- **Per-neuron bias `b_j`**, added after the scale.
- **Activations**: `linear`, `relu`, and numerically stable `softmax`.

# Deviation from the paper's encoding

The block-based encoding here uses **uint16_t column pointers** where the
paper's tighter layout would use uint8. This adds roughly 10 % flash
overhead versus the paper's reported numbers but simplifies indexing
code. The connection-index arrays themselves use uint8 as in the paper
(`BLOCK_SIZE = 256` keeps indices safely in 8-bit range).

## Cross-compiling for Cortex-M0

The engine compiles cleanly with `arm-none-eabi-gcc`:

```bash
arm-none-eabi-gcc \
    -mcpu=cortex-m0 -mthumb -Os \
    -ffunction-sections -fdata-sections \
    -c neuroc.c -o neuroc.o
```

You'll also need a startup file, a linker script, and a way to feed the
input vector in (e.g. from a sensor driver). Those are board-specific and
out of scope for this directory.

## Verifying parity with Keras

The exporter applies the **same** ternary quantization rule used during
training (`threshold = 0.5 * mean(|w|)`) before emitting the index arrays,
so the deployed model and the trained model see identical weights. To
verify end-to-end on a single sample:

```bash
# In Python — what Keras predicts:
python3 -c "
import keras, numpy as np
from c_port.export_model import NeuroCLayer, ternary_quantize
m = keras.models.load_model('keras_examples/mnist_neuroc.keras',
    custom_objects={'NeuroCLayer': NeuroCLayer,
                    'ternary_quantize': ternary_quantize}, compile=False)
(_, _), (x, y) = keras.datasets.mnist.load_data()
sample = x[0].astype('float32').reshape(1, -1) / 255
print('Keras prediction:', m.predict(sample, verbose=0).argmax())
print('True label      :', y[0])
"

# In C — what neuroc_forward predicts:
make run
```

The two predicted classes should match. The full output vectors will agree
to within ~1e-5 (rounding from `%a` float literals + software-float order
of operations).
