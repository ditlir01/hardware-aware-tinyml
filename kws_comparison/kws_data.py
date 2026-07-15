#Google Speech
import os
import ssl
import subprocess
import tarfile
import urllib.request
import numpy as np
import tensorflow as tf

CACHE_DIR = os.path.expanduser("~/.keras/datasets/speech_commands_v01")
TAR_PATH  = os.path.join(CACHE_DIR, "speech_commands_v0.01.tar.gz")
DATA_URL  = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
CACHE_NPZ = os.path.join(CACHE_DIR, "kws_mfcc_10x49_standardized.npz")

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
ACTIVITY_NAMES = KEYWORDS                # alias for consistency with HAR
NUM_CLASSES = len(KEYWORDS)              # 10
INPUT_DIM   = 49 * 10                    # 490 — 49 frames * 10 MFCCs

#Audio/MFCC parameters 
SAMPLE_RATE  = 16000
N_SAMPLES    = 16000  
FRAME_LENGTH = 480    
FRAME_STEP   = 320     
FFT_LENGTH   = 512
N_MEL        = 40
N_MFCC       = 10
MEL_LO_HZ    = 80.0
MEL_HI_HZ    = 7600.0


def _ensure_dataset():
    """Download + extract Speech Commands V1 if not already present."""
    if os.path.isdir(os.path.join(CACHE_DIR, "yes")):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(TAR_PATH):
        print(f"Downloading {DATA_URL} (~1.5 GB) ...")
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(DATA_URL, context=ctx) as r, open(TAR_PATH, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception:
            # Fall back to curl (system trust store)
            subprocess.check_call(["curl", "-fsSL", "-o", TAR_PATH, DATA_URL])
    print("Extracting Speech Commands V1 ...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(CACHE_DIR)


#MFCC pipeline

_MEL_MATRIX = None
def _mel_matrix():
    global _MEL_MATRIX
    if _MEL_MATRIX is None:
        _MEL_MATRIX = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MEL,
            num_spectrogram_bins=FFT_LENGTH // 2 + 1,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=MEL_LO_HZ,
            upper_edge_hertz=MEL_HI_HZ,
        )
    return _MEL_MATRIX


@tf.function
def _wav_to_mfcc(path):
    raw = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(raw, desired_channels=1, desired_samples=N_SAMPLES)
    audio = tf.squeeze(audio, axis=-1)                                   # (16000,)
    stft = tf.signal.stft(audio, frame_length=FRAME_LENGTH,
                          frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
    spec = tf.abs(stft)                                                  # (frames, fft/2+1)
    mel  = tf.matmul(spec, _mel_matrix())                                # (frames, 40)
    log_mel = tf.math.log(mel + 1e-6)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :N_MFCC]  # (49, 10)
    return tf.reshape(mfcc, [-1])                                        # (490,)


#Split + preprocess + cache

def _load_split_lists():
    test_set, val_set = set(), set()
    with open(os.path.join(CACHE_DIR, "testing_list.txt")) as f:
        for line in f:
            test_set.add(line.strip())
    with open(os.path.join(CACHE_DIR, "validation_list.txt")) as f:
        for line in f:
            val_set.add(line.strip())
    return test_set, val_set


def _gather_paths():
    
    test_set, _val_set = _load_split_lists()
    p_train, y_train, p_test, y_test = [], [], [], []
    for cls_idx, kw in enumerate(KEYWORDS):
        kw_dir = os.path.join(CACHE_DIR, kw)
        if not os.path.isdir(kw_dir):
            raise FileNotFoundError(f"missing keyword folder: {kw_dir}")
        for fname in sorted(os.listdir(kw_dir)):
            if not fname.endswith(".wav"):
                continue
            rel = f"{kw}/{fname}"
            abs_path = os.path.join(kw_dir, fname)
            if rel in test_set:
                p_test.append(abs_path); y_test.append(cls_idx)
            else:
                p_train.append(abs_path); y_train.append(cls_idx)
    return p_train, y_train, p_test, y_test


def _mfcc_array(paths, batch_size=256):
    """Compute MFCC for every path via tf.data; return float32 ndarray (N, 490)."""
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_wav_to_mfcc, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    parts = [batch.numpy() for batch in ds]
    return np.concatenate(parts, axis=0).astype("float32")


def _preprocess_to_npz():
    _ensure_dataset()
    p_train, y_train, p_test, y_test = _gather_paths()
    print(f"train+val: {len(p_train)} files, test: {len(p_test)} files")

    print("Computing MFCC features (train+val) ...")
    x_train = _mfcc_array(p_train)
    print(f"  shape: {x_train.shape}")
    print("Computing MFCC features (test) ...")
    x_test = _mfcc_array(p_test)
    print(f"  shape: {x_test.shape}")

    y_train = np.asarray(y_train, dtype=np.int32)
    y_test  = np.asarray(y_test,  dtype=np.int32)

    # Per-feature standardization (mean+std fit on train only, applied to both)
    mean = x_train.mean(axis=0)
    std  = x_train.std(axis=0) + 1e-6
    x_train = ((x_train - mean) / std).astype("float32")
    x_test  = ((x_test  - mean) / std).astype("float32")

    
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    np.savez_compressed(CACHE_NPZ,
                        x_train=x_train, y_train=y_train,
                        x_test=x_test,  y_test=y_test)
    print(f"Saved cache: {CACHE_NPZ}")


def load_kws():
    """Return (x_train, y_train, x_test, y_test) for the 10-keyword KWS task.

    x_*  : float32 (N, 490) standardized MFCC features.
    y_*  : int32   (N,) labels in {0..9}.
    """
    if not os.path.exists(CACHE_NPZ):
        _preprocess_to_npz()
    data = np.load(CACHE_NPZ)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_kws()
    print(f"x_train: {x_train.shape}, dtype={x_train.dtype}")
    print(f"x_test : {x_test.shape}")
    print(f"y_train: {y_train.shape}, classes={np.unique(y_train)}")
    print(f"class balance (train): {dict(zip(KEYWORDS, np.bincount(y_train).tolist()))}")
    print(f"class balance (test) : {dict(zip(KEYWORDS, np.bincount(y_test).tolist()))}")
    print(f"feature range: [{x_train.min():.2f}, {x_train.max():.2f}], mean={x_train.mean():.3f}, std={x_train.std():.3f}")
