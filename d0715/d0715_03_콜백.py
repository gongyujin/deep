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

# 4. 딥러닝 선언 - ANN & DNN (인공신경망)/ CNN (합성곱신경망) / RNN (순환신경망)
model=keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dropout(0.3)) # 30프로 제외 (규제)
model.add(keras.layers.Dense(10,activation="softmax"))
# 옵티마이저 추가
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics='accuracy')

## 콜백
# epochs 20번 반복하면서 가장 손실률이 낮은 시점을 찾아서 model 저장
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5') # 콜백 선언

## 조기종료
# 손실곡선이 해당횟수이상 되면 종료하고 가장 낮은 손실률로 이동저장
# patience : 손실률 상승횟수, restore_best_weights : 가장 낮은 손실률 위치로 이동
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True) # 종기종료 선언


# 5. 딥러닝 훈련
history = model.fit(train_scaled, train_label,epochs=20,\
    validation_data=(val_scaled,val_label),callbacks=[checkpoint_cb,early_stopping_cb])
print(history.history.keys()) # history로 넘어온 변수들 - loss, accuracy, val_loss, val_accuracy
print(history.history['loss'])
print(history.history['accuracy'])

# 멈춘위치를 알려줌
print('epoch 회수 : ',early_stopping_cb.stopped_epoch)

# # 모델저장 - 기울기, 절편만 저장 (가중치만 저장)
# model.save_weights('model_test.h5')
# # 모델불러오기
# model.load_weights('model_test.h5')

# # model 전체 저장
# model.save('model-all.h5')


# 그래프 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()


# 6. 정확도
score=model.evaluate(val_scaled,val_label)
print(score)

