#Simple CIFAR-5 convnet

import numpy as np
import keras
from keras import layers

num_classes = 5
input_shape = (32, 32, 3)   # 32x32 colour image, 3 channels (R, G, B)

# Load CIFAR-10 (the full version with 10 classes) and then filter it down to only the first 5 classes
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# cifar10's labels come as shape (N, 1)2D, flatten them to (N)1D for filtering
y_train = y_train.flatten()
y_test  = y_test.flatten()

# Keep only labels < 5  (airplane=0, automobile=1, bird=2, cat=3, deer=4)
train_mask = y_train < num_classes
test_mask  = y_test  < num_classes
x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test,  y_test  = x_test[test_mask],  y_test[test_mask]

# Scale pixels from [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32") / 255

# CIFAR images already have shape (N, 32, 32, 3) => no expand_dims needed

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0],  "test samples")

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)


# Same architectural pattern as the MNIST convnet, just adapted to the bigger input (32x32x3) and 5 output classes 
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs     = 25   # More than MNIST because CIFAR is a harder task

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)


score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]:.4f}")
