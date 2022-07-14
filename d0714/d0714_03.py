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

# 딥러닝 알고리즘 선택
model=keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))

# 확률적 경사 하강법 : 알고리즘을 훈련시키는 것
# 옵티마이저 : 손실이 제일 적은 알고리즘을 찾는 알고리즘
# # 1. 기본 옵티마이저 SGD확률적경사하강법
# model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')

# # 2. 옵티마이저 분리 SGD확률적경사하강법 - learning_rate=0.01 (default) ; 경사를 내려갈때마다 비율
# 보폭을 줄여주는것
# s=keras.optimizers.SGD(learning_rate=0.01)
# model.compile(optimizer=s,loss='sparse_categorical_crossentropy',metrics='accuracy')

### # 3. momentum (모멘텀과 네스테로브에서 사용)
# 방향을 완화해주는 것
# momentum=keras.optimizers.SGD(momentum=0.5,nesterov=True)
# model.compile(optimizer=momentum,loss='sparse_categorical_crossentropy',metrics='accuracy')

### # 4. adam
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

# -----------------------------------------------------
model.summary()

model.fit(train_scaled,train_label)
score=model.evaluate(test_scaled,test_label)