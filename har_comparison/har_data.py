import os
import ssl
import urllib.request
import zipfile
import numpy as np

CACHE_DIR = os.path.expanduser("~/.keras/datasets/uci-har")
DATA_URL  = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ROOT      = os.path.join(CACHE_DIR, "UCI HAR Dataset")

ACTIVITY_NAMES = ["walking", "walking_upstairs", "walking_downstairs",
                  "sitting", "standing", "laying"]
NUM_CLASSES = 6
INPUT_DIM   = 561


def _download_if_needed():
    if os.path.isdir(ROOT):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, "uci_har.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading UCI HAR Dataset from {DATA_URL} ...")
        try:
            # Use certifi's CA bundle (works on python.org macOS Python).
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(DATA_URL, context=ctx) as r, open(zip_path, "wb") as f:
                f.write(r.read())
        except Exception:
            # Fall back to curl (uses macOS system trust store).
            import subprocess
            subprocess.check_call(["curl", "-fsSL", "-o", zip_path, DATA_URL])
    print("Extracting UCI HAR Dataset ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CACHE_DIR)
    # The outer archive contains an inner "UCI HAR Dataset.zip".
    nested = os.path.join(CACHE_DIR, "UCI HAR Dataset.zip")
    if os.path.exists(nested) and not os.path.isdir(ROOT):
        with zipfile.ZipFile(nested) as z:
            z.extractall(CACHE_DIR)


def load_har():
    _download_if_needed()
    x_train = np.loadtxt(os.path.join(ROOT, "train", "X_train.txt"), dtype="float32")
    y_train = np.loadtxt(os.path.join(ROOT, "train", "y_train.txt"), dtype=int) - 1
    x_test  = np.loadtxt(os.path.join(ROOT, "test",  "X_test.txt"),  dtype="float32")
    y_test  = np.loadtxt(os.path.join(ROOT, "test",  "y_test.txt"),  dtype=int) - 1
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_har()
    print(f"x_train: {x_train.shape}, dtype={x_train.dtype}, range=[{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"x_test : {x_test.shape}")
    print(f"y_train: {y_train.shape}, classes={np.unique(y_train)}")
    print(f"class balance (train): {dict(zip(ACTIVITY_NAMES, np.bincount(y_train).tolist()))}")
    print(f"class balance (test) : {dict(zip(ACTIVITY_NAMES, np.bincount(y_test).tolist()))}")
