# CIFAR5 mlp

import numpy as np
import keras
from keras import layers

#data
num_classes = 5
input_dim   = 32 * 32 * 3

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()

#Keep only labels < 5
train_mask = y_train < num_classes
test_mask  = y_test  < num_classes
x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test,  y_test  = x_test[test_mask],  y_test[test_mask]

#standardization => known mean and std for CIFAR10
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

# Model-bigger mpl than mnist since the dataset is harder
model = keras.Sequential(
    [
        keras.Input(shape=(input_dim,)),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(512,  activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(256,  activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

#training
batch_size = 128
epochs     = 60   #needs more epochs than mnist

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)


#evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]*100:.2f}% vs 63.56% in paper") 

n_params = model.count_params()
print(f"Total params: {n_params:,}")
print(f"At float32: {n_params * 4 / 1024:.1f} KB of weights")
