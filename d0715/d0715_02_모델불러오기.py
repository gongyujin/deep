from lightgbm import train
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기 - keras mnist 파일을 불러오기 : 분리해주는 개념이 아니라 load해주는 개념
(train_data,train_label),(test_data,test_label)=keras.datasets.fashion_mnist.load_data()

# print(type(train_data)) # numpy
# print(train_data.shape) # 필수사항
# print(np.unique(train_label)) # 필수사항

# 2. 정규화, 표준화 작업
train_data=train_data/255
test_data=test_data/255
print(train_data.shape)
print(test_data.shape)
# 3. train, test 데이터 분리
train_scaled,val_scaled,train_label,val_label=train_test_split(train_data,train_label)
print(train_scaled.shape)
print(val_scaled.shape)
print(val_scaled.shape)
#----------------------------------------------------------------------------
# 4. 딥러닝 선언 - ANN & DNN (인공신경망)/ CNN (합성곱신경망) / RNN (순환신경망)
# model=keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28,28)))
# model.add(keras.layers.Dense(100,activation="relu"))
# model.add(keras.layers.Dense(10,activation="softmax"))
# # 옵티마이저 추가
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics='accuracy')

# 모델 불러오기
# model.save - dense나 flatten같은 층이 필요없음
# model.save_weights - 가중치만 저장되기 때문에 층이 필요함
model=keras.models.load_model('best-model.h5') # 전체불러오기
# model=keras.models.load_model('model-all.h5') # 전체불러오기 , dropout이 들어가지 않음
# model.load_weights('model_test.h5') # save_weights 파일 읽어오기, 딥러닝 선언~compile까지 있어야함

# argmax : 최대값의 주소위치 반환
# axis방향 - 열의 방향 val_scaled 개수
val_labels=np.argmax(model.predict(val_scaled),axis=1) # axis=-1이면 3차원일때 제일마지막차원에 붙게됨
# val_labels : 주소값 - 데이터값을 검색해야 함
print(np.mean(val_labels==val_label))
print('-'*50)

# 6. 정확도
score=model.evaluate(val_scaled,val_label)
print(score)

