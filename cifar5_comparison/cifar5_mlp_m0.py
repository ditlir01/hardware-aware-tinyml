# Plain float32 MLP sized for LP-MSPM0G3507 (Cortex-M0+, 128 KB Flash).
#
# CIFAR-5 is the hardest case: 3072-d input means even a tiny hidden layer
# blows past the M0+ flash budget very fast.
#   - 3072->8->5  = 24,621 params * 4 = 99 KB float  (fits)
#   - 3072->16->5 = 49,232 params * 4 = 197 KB float (doesn't fit)
#
# This 3072->8->5 variant is the largest float MLP that fits the M0+. Don't
# expect great accuracy (~45-55%); the 8-neuron bottleneck is brutal.
# That's THE PAPER'S POINT — MLPs barely deploy on M0-class hardware for
# anything harder than digits.

import numpy as np
import keras
from keras import layers


num_classes = 5
input_dim   = 32 * 32 * 3

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()
train_mask = y_train < num_classes
test_mask  = y_test  < num_classes
x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test,  y_test  = x_test[test_mask],  y_test[test_mask]

# Per-channel CIFAR standardization (zero mean / unit std).
mean = np.array([0.4914, 0.4822, 0.4465]) * 255
std  = np.array([0.2470, 0.2435, 0.2616]) * 255
x_train = (x_train.astype("float32") - mean) / std
x_test  = (x_test.astype("float32")  - mean) / std
x_train = x_train.reshape(-1, input_dim)
x_test  = x_test.reshape(-1, input_dim)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)


# 3072*8 + 8 + 8*5 + 5 = 24,621 params * 4 = 99 KB.
model = keras.Sequential([
    keras.Input(shape=(input_dim,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]*100:.2f}%")
print(f"Total params  : {model.count_params():,}")
print(f"Float32 size  : {model.count_params() * 4 / 1024:.1f} KB")

model.save("cifar5_mlp_m0.keras")
print("Saved -> cifar5_mlp_m0.keras")
