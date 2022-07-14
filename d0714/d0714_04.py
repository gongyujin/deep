from tensorflow import keras
import sklearn
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# mnist파일 불러오기
# keras mnist파일이 설치될때 자동설치가 됨
# 튜플형태로 자동적으로 가져오게 됨
(train_data,train_label),(test_data,test_label)=keras.datasets.fashion_mnist.load_data()

train_scaled=train_data/255.0
test_scaled=test_data/255.0

# 딥러닝 선언
model=keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation='softmax'))

# 딥러닝 설정
model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics='accuracy')

# 훈련
history = model.fit(train_scaled,train_label,epochs=20,validation_data=(test_scaled,test_label))
# [loss,accuracy] 출력됨
print(history.history['loss'])
print(history.history['accuracy'])

# plt.plot(history.history['val_accuracy'])
# plt.xlabel('epoch')
# plt.ylabel('val_accuracy')
# plt.show()
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('val_loss')
# plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()