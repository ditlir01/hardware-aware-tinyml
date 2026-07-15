#Simple FashionMNIST convnet
import numpy as np
import keras
from keras import layers

num_classes = 10
input_shape = (28, 28, 1)

# Load Fashion-MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Scale pixels from [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32") / 255

# Conv2D layers expect a channel dimension: (28, 28, 1) for grayscale
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0],  "test samples")

# One-hot encode the labels.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

#Building the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs     = 15

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
