# Reproduction of Neuro-C: Comparison of MLP, TNN, and Neuro-C across MNIST, Fashion-MNIST, and CIFAR-5

*Reference: Romano, Mottola, Voigt. "Neuro-C: Neural Inference Shaped by Hardware Limits", EuroSys '26.*

---

## 1. Scope and goal

The Neuro-C paper compares three model families across three image-classification datasets of increasing difficulty, all targeting deployment on an ARM Cortex-M0-class microcontroller (specifically the STM32F072RB: 128 KB flash, 16 KB RAM, no FPU, no SIMD, no MACC unit).

The three model families the paper compares are:

1. **MLP**: a conventional fully-connected network with full-precision (float32) weights. The "baseline" against which Neuro-C is positioned.
2. **TNN**: a standard ternary neural network, where weights are constrained to `{-1, 0, +1}`, no per-neuron scaling, no normalization. 
3. **Neuro-C**: a TNN augmented with a single per-neuron scaling factor `wⱼ` applied after the ternary adjacency matrix multiplication. Equation 1 of the paper:

   *oⱼ = f( wⱼ · Σᵢ aᵢⱼ · xᵢ + bⱼ )*

   where `aᵢⱼ ∈ {-1, 0, +1}` is the ternary adjacency matrix, `wⱼ` is the per-neuron scale, `bⱼ` is the per-neuron bias, and `f` is the activation function.

Across all three families the paper uses **fully-connected layers, never convolutional layers**, as a deliberate architectural choice (Sec 3.3), since convolutions require `im2col` reshaping and irregular memory access patterns that map poorly onto a Cortex-M0 with no SIMD or DSP support.

The repository reproduces all nine configurations (3 model families × 3 datasets), organized as three sibling folders: `mnist_comparison/`, `fmnist_comparison/`, `cifar5_comparison/`. Each folder contains three scripts: `*_mlp.py`, `*_tnn.py`, and `*_neuroc_paper.py`.

Beyond the paper's image reproduction, the thesis extends Neuro-C to **time-series IMU/gesture data** via the UCI HAR Dataset (`har_comparison/`). It follows the same MLP / TNN / Neuro-C structure with a shared `har_data.py` loader that encodes the temporal pipeline (2.56 s sliding windows + 561 handcrafted time/frequency features per Anguita 2013).This contribution wants to demonstrate that the hardware-aware ternary-inference principles introduced for static images also work on sequential IMU sensor data.

---

## 2. Architectural decisions and their justifications

### 2.1 Why fully-connected, not convolutional

The original Keras tutorial that inspired this repository used Conv2D layers. I deliberately moved every architecture in this comparison to a flattened-input FC pipeline. The justification comes directly from the paper:

> *"For an FC layer with `Nᵢₙ` input neurons and `Nₒᵤₜ` output neurons, each output is computed as a weighted sum of all input neurons. The number of MACC operations is `MACCs_FC = Nₒᵤₜ × Nᵢₙ`. [For a CNN] `MACCs_CNN = K · C · S² · M²`. ... For the Cortex-M0, FC layers exhibit lower inference latency due to simpler memory accesses and control flows, regardless of dimension."* (Sec 3.3)

Convolutions need `im2col` matrix reshaping and strided indexing, both of which require either SIMD acceleration or explicit loop machinery on the MCU. FC layers map to one pointer walk per neuron, making the access pattern predictable, cache-friendly (insofar as the M0 has any cache), and fixed control flow.

For comparison purposes I kept the original CNN files in the repo as well, but the three "family" comparisons (`mlp` / `tnn` / `neuroc_paper`) are all FC.

### 2.2 Why these specific layer widths and depths

The paper does **not** publish exact layer dimensions. Section 5.2 states they perform "manual model selection across various configurations" and "extensive random search over more than 50 MLP configurations". They report only the best deployable configuration per dataset, constrained to ~20–35 KB of program memory.

This meant I had to choose architectures myself. My rationale per dataset:

**MNIST (`mnist_comparison/`):**
- MLP: `784 → 512 → 256 → 128 → 10` with dropout 0.2. Three hidden layers because that's the classic depth that hits ~98.3% on MNIST (matching the paper's target).
- Neuro-C paper version: `784 → 256 → 128 → 64 → 10`. Smaller than the MLP because ternary weights compress 20× and `wⱼ` adds expressivity, so I need less width per layer to hit ~99% accuracy at a deployable size.
- TNN: same depth/widths as the Neuro-C paper version (Sec 5.2: *"remaining architecture and training protocol are kept identical"*). When this proved unstable, I shrank to a single hidden layer.

**Fashion-MNIST (`fmnist_comparison/`):**
- Same architectures as MNIST. Fashion-MNIST is the same shape (28×28×1, 10 classes) but harder (clothing items overlap visually more than digits). I increased dropout from 0.2 to 0.3 in the MLP to compensate for the increased overfitting risk, and bumped MLP epochs from 30 → 40.

**CIFAR-5 (`cifar5_comparison/`):**
- MLP: `3072 → 1024 → 512 → 256 → 5` with dropout 0.4. Bigger network because the input is ~4× larger (3072-d vs 784-d) and CIFAR is much harder than MNIST. Aggressive dropout because CIFAR-5 has fewer training examples after class-filtering (~25K vs 60K).
- Neuro-C paper version: `3072 → 512 → 256 → 128 → 64 → 32 → 16 → 5`. Deeper than the MNIST version because CIFAR-5 needs more representational depth, and the per-neuron scale `wⱼ` makes deep ternary stacks trainable.
- TNN: same architecture as Neuro-C on CIFAR-5, with `wⱼ` removed and (as expected from the paper) this fails to converge.

### 2.3 Why no batch normalization

The paper rejects batch norm because it cannot be folded into ternary weights (Sec 3.4):

> *"Existing ternary networks rely on per-layer scaling and batch normalization. With fixed quantized weights, batch normalization cannot be folded into the model and must be computed at runtime, adding substantial overhead that low-end MCUs cannot afford."*

Per-neuron scaling `wⱼ` in Neuro-C serves the same stabilizing role as batch norm during training, but it's **just a single multiply per output neuron** at deployment. Batch norm would require a multiply, an add, a square root, and a division per activation, which is unaffordable.

So none of my scripts use `BatchNormalization`, for MLP either (to keep memory footprints honest in the comparison).

### 2.4 Why centered or standardized inputs

For MNIST I use `x / 255` to get `[0, 1]`, which is the standard pixel scaling. For Fashion-MNIST TNN I use `x / 255 - 0.5` to get `[-0.5, 0.5]`, because symmetric inputs work better with ±1 ternary weights because there's no built-in positive bias from one-sided inputs. For CIFAR-5 I use full per-channel standardization (zero mean, unit std) using the canonical CIFAR-10 channel statistics:

```python
mean = np.array([0.4914, 0.4822, 0.4465]) * 255
std  = np.array([0.2470, 0.2435, 0.2616]) * 255
x = (x - mean) / std
```

Why the escalation? With ternary weights `±1` and a 3072-d input, the first-layer pre-activation sum has magnitude `~sqrt(3072) ≈ 55` if inputs are uncentered. That magnitude blows up softmax and Adam destroys the network in the first few epochs. Centering and normalizing keeps the sum near unit variance, which is the only range ternary weights can sensibly operate in.

For HAR the UCI HAR Dataset already ships the 561 handcrafted features pre-normalised to `[-1, 1]` (Anguita 2013), so no additional centering is needed. The same conditions that make ternary nets stable on standardised CIFAR inputs hold for HAR by construction.

---

## 3. Implementation details

### 3.1 Hand-rolled ternary quantization with straight-through estimator (STE)

Every TNN and Neuro-C layer in the repository uses this quantization function:

```python
def ternary_quantize(latent):
    threshold = 0.5 * tf.reduce_mean(tf.abs(latent))
    q = tf.zeros_like(latent)
    q = tf.where(latent >  threshold,  tf.ones_like(latent), q)
    q = tf.where(latent < -threshold, -tf.ones_like(latent), q)
    latent_clipped = tf.clip_by_value(latent, -1.0, 1.0)
    return tf.stop_gradient(q - latent_clipped) + latent_clipped
```

Two design choices:

1. **TWN-style adaptive threshold** (`0.5 * mean(|w|)`) rather than a fixed threshold. The threshold adapts to the weight distribution: at initialization, weights are small (glorot stddev ≈ 0.04 for an 784×512 layer), so the threshold is also small, keeping the ternary matrix sparse but not empty. As training progresses and weights spread out, the threshold grows accordingly. I tested a fixed threshold of `0.05` (Larq's default) on Fashion-MNIST and accuracy crashed from 78% to 60% because the threshold was above the initial weight magnitudes. Almost every weight quantized to zero and the network had no connectivity to learn through.

2. **Straight-through estimator** for the backward pass. The line:
   ```python
   return tf.stop_gradient(q - latent_clipped) + latent_clipped
   ```
   evaluates to `q` in the forward pass (because `q - latent_clipped + latent_clipped = q`), but `tf.stop_gradient` makes the backward pass see only `latent_clipped`. So forward = discrete ternary, backward = gradient through `clip(latent, -1, 1)`. 

### 3.2 The Larq question

The paper trains all its models via the Larq framework (Sec 5.2), which provides ternary-aware initialization, gradient clipping, and other stabilization tricks. **I cannot use Larq in this reproduction** because Larq is not compatible with Keras 3 (the version that ships with TensorFlow ≥ 2.16). Attempting to import Larq fails with `AttributeError: module 'keras._tf_keras.keras.layers' has no attribute 'LocallyConnected1D'` because Keras 3 removed `LocallyConnected1D`, which Larq's `__init__.py` still imports.

Two options were available:
1. Downgrade TensorFlow to `<2.16`, which might break other parts of the environment.
2. Replicate Larq's most important trick by hand.

I chose option 2. Larq's single most important stabilization is a `weight_clip` constraint that clips latent weights to `[-1, 1]` after every Adam step. I implemented this as a four line custom constraint:

```python
class WeightClip(keras.constraints.Constraint):
    def __init__(self, clip_value=1.0):
        self.clip_value = clip_value
    def __call__(self, w):
        return tf.clip_by_value(w, -self.clip_value, self.clip_value)
    def get_config(self):
        return {"clip_value": self.clip_value}
```

**Why this matters:** the STE backward pass passes gradient only through `clip(latent, -1, 1)`. If a latent weight drifts to magnitude 2.5 during training (which Adam will do without constraint), its STE gradient is **zero**, the weight freezes in place forever. The constraint pulls drifting weights back into `[-1, 1]` after every step, keeping STE gradients alive. Without it, deep ternary networks stall during training.

This trick is the difference between a Fashion-MNIST TNN that reaches ~80% and one that stalls at ~50%.

### 3.3 Inevitable Larq accuracy gap

Even with `WeightClip`, my hand-rolled STE leaks a few percentage points to a properly Larq-trained version. The gap is structural, not a tuning problem. Larq does several other small things (ternary-aware initialization, gradient handling at the quantization boundary, specific weight regularization) that together account for roughly 3–5 percentage points of accuracy on Fashion-MNIST and CIFAR-5. On MNIST (where the task is easy enough that the network finds the right minimum regardless) the gap is smaller.

I report this gap honestly in the results section.

---

## 4. Results and what each one demonstrates

### 4.1 The headline comparison table

| Dataset | MLP (float32) | TNN (ternary, no `wⱼ`) | Neuro-C (ternary + `wⱼ`) | Paper Neuro-C target |
|---|---|---|---|---|
| MNIST | 98,40% | 96,05% | 98,30% | 99,10% |
| Fashion-MNIST | 89,23% | 81,01% | 88,53% | 89,90% |
| CIFAR-5 | 67,08% | **no convergence** | 67,12% | 67.58% |

### 4.2 The Fashion-MNIST TNN troubleshooting story

**Initial attempt (3 hidden layers, no `WeightClip`):** 11% accuracy. Random for a 10-class problem. The network was making confident wrong predictions and gradients had vanished.

**Examination:** Without per-neuron scale `wⱼ` and without batch normalization, the pre-activation magnitudes compound through ReLU layers:
- Input range: `[0, 1]`
- Layer 1 output range: `~[0, 3]` after ReLU
- Layer 2 output range: `~[0, 20]` after ReLU
- Layer 3 output range: `~[0, 100]` after ReLU
- Output layer logits: magnitude `~±500`
- Softmax saturates → gradients vanish → Adam has nothing to optimize → training stalls

**This is exactly the paper's claim in Sec 5.2:**
> *"When the per-neuron scaling factor wⱼ is removed ... the model fails to converge on CIFAR5 and suffers accuracy drops of 2.5 and 3.5 percentage points on MNIST and FashionMNIST."*

The paper still achieves 86.35% Fashion-MNIST TNN accuracy because **Larq's training infrastructure prevents the magnitude compounding** through specific initialization, gradient handling, and weight clipping.

**Final Fashion-MNIST TNN setup:** single hidden layer of 1024 units, TWN-style adaptive threshold, `WeightClip(1.0)`, inputs centered to `[-0.5, 0.5]`, Adam at LR `5e-4`, 100 epochs without early stopping.

**Final result: 81.56% test accuracy** (vs paper's 86.35%). The ~5 pp gap is the cost of not using Larq.

### 4.3 The CIFAR-5 TNN: an intentional failure

The CIFAR-5 TNN file is **supposed to fail to converge**. This is the paper's most dramatic finding (Sec 5.2):

> *"This configuration fails to converge entirely on CIFAR5, providing evidence of the role of wⱼ in stabilizing the training dynamics as the input complexity increases."*

The expected pattern: loss hovers around `log(5) ≈ 1.61` (random-guess loss for 5 classes), accuracy stays around 20%, and training never improves. 


### 4.4 HAR / IMU extension

This is the part of the project specific to the thesis rather than the paper reproduction. The question is: do the Neuro-C principles, ternary connectivity, per-neuron scaling, no MAC operations, actually transfer outside the image domain? Specifically, do they survive on **time-series IMU sensor data**, which is what wearables and IoT motion-recognition systems actually consume?

**Temporal pipeline.** I use the standard UCI HAR pipeline (Anguita 2013): 2.56 s sliding windows over the raw 9-channel IMU (3-axis body acceleration + 3-axis total acceleration + 3-axis gyroscope, sampled at 50 Hz) with 50% overlap, then **561 handcrafted features per window**, time-domain statistics (mean, std, MAD, max, min, SMA, energy, IQR, AR coefficients, inter-axis correlations) and frequency-domain features (FFT magnitudes per band, dominant/mean frequency, skewness, kurtosis, spectral entropy). The result is a fixed-length 561-d input vector per window, already normalised to `[-1, 1]`. 

**Headline scoreboard** (single run, UCI HAR, 7352 train / 2947 test, 6 activities; the test set holds out specific subjects → cross-subject generalization):

| Model | Architecture | Params | Test acc | Deployable size |
|---|---|---|---|---|
| MLP (float32) | `561 → 256 → 128 → 64 → 6` | 185,414 | 94.13% | ~724 KB (4× over M0 budget) |
| **Neuro-C** | `561 → 256 → 128 → 64 → 6` | 185,868 latent | **94.40%** | ~50 KB ternary + scales |
| TNN (1 hidden) | `561 → 256 → 6` | 145,414 | 92.50% | — |

**What this demonstrates:**

1. **Neuro-C matches/exceeds the MLP on real IMU data** (94.40% vs 94.13% at matched architecture).

2. **The size gap is dramatic at matched architecture.** Both nets carry ~185K parameters; at deployment the MLP needs ~724 KB float32 (4× over the 128 KB M0 flash budget — non-deployable), while Neuro-C's ternary adjacency plus per-neuron scales pack into ~50 KB. Roughly 15× smaller.

---

## 8. Repository layout

```
Neuro-C-internship/
├── README.md                      project overview
├── mnist_comparison/              MNIST: all three model families
│   ├── mnist_mlp.py
│   ├── mnist_tnn.py
│   └── mnist_neuroc_paper.py
├── fmnist_comparison/             Fashion-MNIST: same three
│   ├── fmnist_mlp.py
│   ├── fmnist_tnn.py
│   └── fmnist_neuroc_paper.py
├── cifar5_comparison/             CIFAR-5: same three
│   ├── cifar5_mlp.py
│   ├── cifar5_tnn.py
│   └── cifar5_neuroc_paper.py
├── har_comparison/                UCI HAR (IMU/gesture): thesis extension
│   ├── har_data.py                shared loader + temporal pipeline (Anguita 2013)
│   ├── har_mlp.py
│   ├── har_tnn.py
│   └── har_neuroc_paper.py
├── keras_examples/                additional experiments
│   ├── mnist_neuroc.py            single-hidden-layer Neuro-C (original)
│   ├── mnist_neuroc_metrics.py    Neuro-C with Sec 5.1 metric instrumentation
│   ├── har_neuroc_paper.py        early HAR draft (superseded by har_comparison/)
│   └── kws_neuroc_paper.py        Google Speech Commands (KWS) extension
└── c_port/                        C inference engine for MCU deployment
    ├── neuroc.h
    ├── neuroc.c
    ├── export_model.py            converts trained .keras → C header
    ├── example_main.c
    └── Makefile
```

Each subfolder is independently runnable: `cd mnist_comparison && python3 mnist_neuroc_paper.py`.

