from gc import callbacks
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb

# 파일불러오기
# 웹스크래핑 불러오기 : 500개 넘버링만 가져오기
(train_data,train_label),(test_data,test_label)=imdb.load_data(num_words=500)
# (25000,) 1차원배열
print(train_data.shape,test_data.shape)

## 단어의 수 : 218개 예) 영화 가 너무 재미 없음. 짱 재미 없음. 정말 재미 ... => 번호처리 해줌
## [1, 14, 22, 16, 43, 2, 2, 2, 2, => 2는 500단어에서 없는 것을 의미함, 500번대 이상은 없음
# print(train_data[0])
# print(len(train_data[0])) 

# [0, 1]
print(train_label)
print(np.unique(train_label))


# -----------------------------------------------------------------------
# 데이터 전처리
sub_data,val_data,sub_label,val_label=train_test_split(train_data,train_label)

# (18750,) (6250,)
print(sub_data.shape,val_data.shape)

# 각 train_data의 문장길이가 어떻게 되는지 확인
# 25000개의 단어길이 합을 구함
lengths=np.array([len(x) for x in train_data])

# 평균값 : 238.71364, 중간값 : 178.0
print(np.mean(lengths),np.median(lengths)) # 더 낮은 값으로 채우는게 확률적으로 더 좋음
print(np.max(lengths)) # 최대 단어길이 : 2494
print(np.min(lengths)) # 최소 단어길이 : 11

# 그래프 그리기
plt.hist(lengths)
plt.xlabel('lengths')
plt.ylabel('frequency') # hist y value값 다 합치면 아마도 25000개
plt.show()

# ----------------------------------------------------------------------
# 11 ~ 2494 글자 존재 => 100글자만 사용할 것
# 100글자 짜르고, 없는 부분 0으로 채워줌
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 원래 문장길이가 11 ~ 2494개 중 하나인데 maxlen으로 100개로 맞춰줌
# train_data -> sub_data
train_seq=pad_sequences(sub_data,maxlen=100)
# 문장길이 218 -> 문장길이 100으로 변경, 문장길이 100 나머지 0으로 채움
print(train_seq[0])
print(train_seq[5])

# test_data -> val_data
test_seq=pad_sequences(val_data,maxlen=100)

# 문장길이 : 100, 단어개수 : 500    => (100,500) ; 500개의 단위로 원핫인코딩이 들어감 500/500/500/500 ....
# 312 489  10  10   2  47  69   2  11   2  11   2 153  19   2 245   5   2
# => 312개를 500개 단위 원핫인코딩으로 만들어짐
####### 원핫인코딩
train_oh=keras.utils.to_categorical(train_seq)
print(train_oh[0]) # (18750, 100, 500)

test_oh=keras.utils.to_categorical(test_seq)
print(test_oh[0]) # (18750, 100, 500)


# 순환 신경망 선언
model=keras.Sequential()
# 순환 신경망 - 뉴런개수:8
model.add(keras.layers.SimpleRNN(8,input_shape=(100,500)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

# ----------------------------------------------------
# 순환신경망 설정 adam, RMSprop
rmsprop=keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics='accuracy')

# 콜백 - 20번돌고나서, 가장 낮은 손실률을 저장
check_cb=keras.callbacks.ModelCheckpoint('best-rnn.h5',save_best_only=True)
early_cb=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)

history=model.fit(train_oh,sub_label,epochs=100,batch_size=64,\
    validation_data=(test_oh,val_label),callbacks=[check_cb,early_cb])

# 그래프 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.show()

# -------------------------------------------------------
# 정확도
score= model.evaluate(test_oh,val_label)
print('loss, accuracy : ', score)
