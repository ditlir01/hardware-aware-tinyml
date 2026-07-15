#fmnist tnn
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


class WeightClip(keras.constraints.Constraint): #custom constraint class to clip latent weights to [-1, 1]
    def __init__(self, clip_value=1.0):
        self.clip_value = clip_value
    def __call__(self, w):
        return tf.clip_by_value(w, -self.clip_value, self.clip_value)
    def get_config(self):
        return {"clip_value": self.clip_value}


#tnn layer
class TNNLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.latent_kernel = self.add_weight(
            name="latent_kernel", shape=(in_dim, self.units),
            initializer="glorot_uniform",
            constraint=WeightClip(1.0),         # keeps ste gradient alive
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,),
            initializer="zeros", trainable=True,
        )

    def call(self, x):
        A = ternary_quantize(self.latent_kernel)
        z = tf.matmul(x, A) + self.bias
        return self.activation(z)

num_classes = 10
input_dim   = 28 * 28

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255 - 0.5 #0.5 is the mean of the original distribution, so this just centers it on zero
x_test  = x_test.astype("float32")  / 255 - 0.5 
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
        TNNLayer(1024, activation="relu"),#two hidden layers without scaling factor is unstable
        TNNLayer(num_classes, activation="softmax"),
    ]
)

model.summary()

#training
batch_size = 128
epochs     = 100

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=5e-4), ##lr sensitivity is worse for deeper TNNs; 5e-4 works reliably for 2 layers, but 1e-3 causes divergence in some runs
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

#evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]*100:.2f}% vs 86.35 in paper")


