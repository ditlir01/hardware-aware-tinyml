# NeuroC  CIFAR5 

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
        self.units      = units
        self.activation = keras.activations.get(activation)
        self.scale_init = scale_init   # None -> w_j = 1/sqrt(fan_in)

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
            initializer=scale_initializer,
            trainable=True,
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

num_classes = 5
input_dim   = 32 * 32 * 3      #3072 features

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()

# Keep only labels < 5 (airplane, automobile, bird, cat, deer)
train_mask = y_train < num_classes
test_mask  = y_test  < num_classes
x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test,  y_test  = x_test[test_mask],  y_test[test_mask]

#per channel standardization (zero mean, unit std)
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


#building model
model = keras.Sequential(
    [
        keras.Input(shape=(input_dim,)),
        NeuroCLayer(512, activation="relu"),
        NeuroCLayer(256, activation="relu"),
        NeuroCLayer(128, activation="relu"),
        NeuroCLayer(64,  activation="relu"),
        NeuroCLayer(32,  activation="relu"),
        NeuroCLayer(16,  activation="relu"),
        NeuroCLayer(num_classes, activation="softmax"),
    ]
)

model.summary()


#train
batch_size = 128
epochs     = 500

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)


#evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")
print(f"Paper target: 0.6758")


print()
print("Adjacency matrices ")
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
