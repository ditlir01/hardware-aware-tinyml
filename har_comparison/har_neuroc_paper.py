# HAR Neuro-C

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from har_data import load_har, ACTIVITY_NAMES, NUM_CLASSES, INPUT_DIM
 
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
        self.units      = units
        self.activation = keras.activations.get(activation)
        self.scale_init = scale_init

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
            initializer=scale_initializer, trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,),
            initializer="zeros", trainable=True,
        )

    def call(self, x):
        A = ternary_quantize(self.latent_kernel)   
        z = tf.matmul(x, A)                        
        z = self.scale * z + self.bias             


x_train, y_train, x_test, y_test = load_har()
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")


model = keras.Sequential([
    keras.Input(shape=(INPUT_DIM,)),
    NeuroCLayer(512, activation="relu"),
   NeuroCLayer(256, activation="relu"),
   NeuroCLayer(128, activation="relu"),
    NeuroCLayer(64,  activation="relu"),
    NeuroCLayer(32,  activation="relu"),
   NeuroCLayer(16,  activation="relu"),
    NeuroCLayer(NUM_CLASSES, activation="softmax"),
])
model.summary()

batch_size = 128
epochs     = 200  

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train_oh,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

score = model.evaluate(x_test, y_test_oh, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")


print()
print("Adjacency matrices (post-quantization)")
for i, layer in enumerate(model.layers):
    if not isinstance(layer, NeuroCLayer):
        continue
    A = ternary_quantize(layer.latent_kernel).numpy()
    total = A.size
    zeros = int(np.sum(A ==  0))
    pos   = int(np.sum(A == +1))
    neg   = int(np.sum(A == -1))
    print(f"Layer {i}  shape={A.shape}  "
          f"0={zeros/total:.1%}  +1={pos/total:.1%}  -1={neg/total:.1%}")
