# CIFAR5 TNN
import numpy as np
import tensorflow as tf
import keras
from keras import layers


#ternary quantization
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


#data (first 5 classes)
num_classes = 5
input_dim   = 32 * 32 * 3

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()
train_mask = y_train < num_classes
test_mask  = y_test  < num_classes
x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test,  y_test  = x_test[test_mask],  y_test[test_mask]

#standardization per channel
mean = np.array([0.4914, 0.4822, 0.4465]) * 255
std  = np.array([0.2470, 0.2435, 0.2616]) * 255
x_train = (x_train.astype("float32") - mean) / std
x_test  = (x_test.astype("float32")  - mean) / std

x_train = x_train.reshape(-1, input_dim)
x_test  = x_test.reshape(-1, input_dim)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0],  "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)


# building model
model = keras.Sequential(
    [
        keras.Input(shape=(input_dim,)),
        TNNLayer(512, activation="relu"),
        TNNLayer(256, activation="relu"),
        TNNLayer(128, activation="relu"),
        TNNLayer(num_classes, activation="softmax"),
    ]
)

model.summary()


#training
batch_size = 128
epochs     = 50

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)


#evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]*100:.2f}%")
print(f"Paper result  : NO CONVERGENCE")
