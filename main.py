import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

dataset = tf.keras.datasets.mnist

## Loading Dataset

(X_train, y_train), (X_test, y_test) = dataset.load_data()

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# CV2 Image Show
# cv2.imshow('image', X_train[20])
# cv2.waitKey(0)

## Normalization Value to b/w 0~1

X_train = X_train/255.0
X_test = X_test/255.0
# train, test dataset shape : (, 28, 28)

## CNN Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input

inputs = Input(shape=[28, 28, 1])

conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu')(inputs)
conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(conv1)

flatten = Flatten()(conv2)

dense2 = Dense(10, activation='softmax')(flatten)

model = tf.keras.models.Model(inputs, dense2)
model.compile('adam', 'categorical_crossentropy', metrics=['acc'])

model.fit(X_train, y_train, batch_size=16, epochs=10)

## Using cv2 Library for Video Capture

model.save('model.h5')

