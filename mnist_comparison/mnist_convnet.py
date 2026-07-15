#Simple MNIST convnet

import numpy as np 
import keras
from keras import layers

# MNIST has 10 classes (the digits 0..9) and each image is 28x28 grayscale.
num_classes = 10 # MNIST is a handwritten digit dataset. The digits are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 → 10 possible answers
input_shape = (28, 28, 1) #The input images are 28 pixels wide, 28 pixels high, and have 1 color channel (grayscale).

# Loadind the data: keras downloads it the first time. [MNIST is built into keras, so no need to download it separately. Keras will handle that for me]
# x = the images (input),  y = the labels (correct digit 0..9), train = the training set, 60.000 (used to train the model), test = the test set, 10.000 (used to evaluate the model after training)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale pixel values from the integer range [0, 255] to the float range [0, 1].
# Neural networks train much better when inputs are small numbers around 0.
x_train = x_train.astype("float32") / 255 
x_test  = x_test.astype("float32") / 255

# Conv2D layers expect images with a channel dimension: (28, 28, 1).
# MNIST is grayscale so there is only 1 channel.
x_train = np.expand_dims(x_train, -1) 
x_test  = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0],  "test samples")

# Convert the labels from integers (e.g. 3) to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

# building the model
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

#compile the model: specify the loss function, the optimizer, and the metrics to track during training
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

#training the model: feed the training data to the model and specify the batch size, number of epochs, and 
#validation split (10% of the training data will be used for validation during training)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

#evaluate the model on the test set to see how well it generalizes to new data
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]:.4f}")
