# MNIST TNN
import numpy as np
import tensorflow as tf
import keras
from keras import layers

#Ternary quantization
def ternary_quantize(latent):
    threshold = 0.5 * tf.reduce_mean(tf.abs(latent))
    q = tf.zeros_like(latent)
    q = tf.where(latent >  threshold,  tf.ones_like(latent), q)
    q = tf.where(latent < -threshold, -tf.ones_like(latent), q)
    latent_clipped = tf.clip_by_value(latent, -1.0, 1.0)
    return tf.stop_gradient(q - latent_clipped) + latent_clipped



#TNN layer => no scaling factor (difference from NeuroC)
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


#data
num_classes = 10
input_dim   = 28 * 28

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32")  / 255
x_train = x_train.reshape(-1, input_dim)
x_test  = x_test.reshape(-1, input_dim)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0],  "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

#building the model
model = keras.Sequential(
    [
        keras.Input(shape=(input_dim,)),
        TNNLayer(256, activation="relu"),
        TNNLayer(num_classes, activation="softmax"),
    ]
)

model.summary()

#training
batch_size = 128
epochs     = 50

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)
#evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]*100:.2f}%")
