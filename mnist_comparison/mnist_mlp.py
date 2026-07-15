#MNIST MLP

import numpy as np
import keras
from keras import layers

# Data (flattening data)
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
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

#training
batch_size = 128
epochs     = 30

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

#Evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]*100:.2f}% vs 98.34 in paper")


#Parameter count
n_params = model.count_params()
print(f"Total params: {n_params:,}")
print(f"At float32: {n_params * 4 / 1024:.1f} KB of weights") #2.2MB => MLP doesn't fit on the MCU
