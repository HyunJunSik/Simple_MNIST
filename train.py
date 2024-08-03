import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input

# MNIST 데이터셋 Loading
dataset = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = dataset.load_data()

# 불러온 이미지 보기
img = X_train[49]
img = cv2.resize(img, (500, 500))
cv2.imshow('image', img)
cv2.waitKey(0)

# 데이터 전처리
# Multi-Class Classification임에 따라 One-Hot Encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 데이터셋 차원 맞추기 위한 4차원 변환
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 모델 구성
inputs = Input(shape=[28, 28, 1])
conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu')(inputs)
conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(conv1)
flatten = Flatten()(conv2)
dense1 = Dense(10, activation='softmax')(flatten)

model = tf.keras.models.Model(inputs, dense1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# 모델 학습
model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test))

# 모델 저장
model.save('model.h5')