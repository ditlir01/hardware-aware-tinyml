# Plain float32 MLP sized for LP-MSPM0G3507 (Cortex-M0+, 128 KB Flash).
# Single hidden layer 24 units; ~19K params × 4 bytes = ~77 KB weights.
# Plus ~20 KB code/runtime/printf = total ~97 KB. Fits 128 KB budget.
# Expected accuracy ~96% (slightly below the MSP432 MLP because narrower).

import numpy as np
import keras
from keras import layers


num_classes = 10
input_dim = 28 * 28
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32") / 255
x_train = x_train.reshape(-1, input_dim)
x_test  = x_test.reshape(-1, input_dim)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)


# 784*24 + 24 + 24*10 + 10 = 19,114 params * 4 = 76 KB
model = keras.Sequential([
    keras.Input(shape=(input_dim,)),
    layers.Dense(24, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]*100:.2f}%")
print(f"Total params  : {model.count_params():,}")
print(f"Float32 size  : {model.count_params() * 4 / 1024:.1f} KB")

model.save("mnist_mlp_m0.keras")
print("Saved -> mnist_mlp_m0.keras")
