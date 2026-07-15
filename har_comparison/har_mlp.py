# HAR MLP baseline

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import keras
from keras import layers
from har_data import load_har, ACTIVITY_NAMES, NUM_CLASSES, INPUT_DIM



x_train, y_train, x_test, y_test = load_har()
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")



model = keras.Sequential([
    keras.Input(shape=(INPUT_DIM,)),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64,  activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32,  activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16,  activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])
model.summary()


batch_size = 128
epochs     = 100

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train_oh,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)



#Evaluation 
score = model.evaluate(x_test, y_test_oh, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]*100:.2f}%")

n_params = model.count_params()
print(f"Total params: {n_params:,}")
print(f"At float32: {n_params * 4 / 1024:.1f} KB of weights")

