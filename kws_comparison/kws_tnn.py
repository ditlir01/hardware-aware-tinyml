# KWS (Google Speech Commands V1)TNN baseline

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from kws_data import load_kws, ACTIVITY_NAMES, NUM_CLASSES, INPUT_DIM


def ternary_quantize(latent):
    threshold = 0.5 * tf.reduce_mean(tf.abs(latent))
    q = tf.zeros_like(latent)
    q = tf.where(latent >  threshold,  tf.ones_like(latent), q)
    q = tf.where(latent < -threshold, -tf.ones_like(latent), q)
    latent_clipped = tf.clip_by_value(latent, -1.0, 1.0)
    return tf.stop_gradient(q - latent_clipped) + latent_clipped


class TNNLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.latent_kernel = self.add_weight(
            name="latent_kernel", shape=(in_dim, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,),
            initializer="zeros", trainable=True,
        )

    def call(self, x):
        A = ternary_quantize(self.latent_kernel)
        z = tf.matmul(x, A) + self.bias
        return self.activation(z)
    
x_train, y_train, x_test, y_test = load_kws()
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")

model = keras.Sequential([
    keras.Input(shape=(INPUT_DIM,)),
    TNNLayer(512, activation="relu"),
     TNNLayer(256, activation="relu"),
    TNNLayer(NUM_CLASSES, activation="softmax"),
])
model.summary()

batch_size = 128
epochs     = 100

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train_oh,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

score = model.evaluate(x_test, y_test_oh, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]*100:.2f}%")
print(f"Note          : single-layer literal TNN, no w_j. Compare to Neuro-C on")
print(f"                the same input for the per-neuron-scale ablation.")
