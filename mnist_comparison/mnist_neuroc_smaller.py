# Tighter Neuro-C MNIST sized to approach paper's ~20 KB deployment footprint.
# Architecture: 784 -> 32 -> 16 -> 10 (half the widths of the nucleo variant).
# Expected accuracy: ~94-95% (vs nucleo variant's 96.83%).
# Expected deployed footprint: ~25-30 KB total (model data ~12-15 KB).

import numpy as np
import tensorflow as tf
import keras
from keras import layers


def ternary_quantize(latent):
    threshold = 0.5 * tf.reduce_mean(tf.abs(latent))
    q = tf.zeros_like(latent)
    q = tf.where(latent >  threshold,  tf.ones_like(latent), q)
    q = tf.where(latent < -threshold, -tf.ones_like(latent), q)
    latent_clipped = tf.clip_by_value(latent, -1.0, 1.0)
    return tf.stop_gradient(q - latent_clipped) + latent_clipped


class NeuroCLayer(layers.Layer):
    def __init__(self, units, activation=None, scale_init=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.scale_init = scale_init

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.latent_kernel = self.add_weight(
            name="latent_kernel", shape=(in_dim, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        if self.scale_init is None:
            init = keras.initializers.Constant(1.0 / np.sqrt(in_dim))
        else:
            init = keras.initializers.RandomUniform(*self.scale_init)
        self.scale = self.add_weight(
            name="scale", shape=(self.units,), initializer=init, trainable=True,
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


num_classes = 10
input_dim = 28 * 28
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32")  / 255
x_train = x_train.reshape(-1, input_dim)
x_test  = x_test.reshape(-1, input_dim)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)


# Smaller architecture — half the widths of the nucleo variant.
model = keras.Sequential([
    keras.Input(shape=(input_dim,)),
    NeuroCLayer(32, activation="relu"),
    NeuroCLayer(16, activation="relu"),
    NeuroCLayer(num_classes, activation="softmax"),
])
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]*100:.2f}%")

model.save("mnist_neuroc_smaller.keras")
print("Saved -> mnist_neuroc_smaller.keras")
