from tensorflow import keras
import sklearn
from sklearn.linear_model import SGDClassifier # 확률적경사하강법
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# train: train.csv - label분리, train,test데이터 분리
# test: t10k.csv - label분리, train,test데이터 분리
# 딥러닝을 사용해서 정확도를 출력하시오.
# 데이터 불러오기
train=pd.read_csv('11.deep/d0714/train.csv',header=None)
test=pd.read_csv('11.deep/d0714/t10k.csv',header=None)


## 데이터 전처리

# 함수사용해서 0~1의 값으로 변경
# def func(a):
#     output=[]
#     for i in a:
#         output.append(float(i)/256)
#     return output
# # data만 map으로 재분배해줌
# train_data=list(map(func,train_csv.iloc[:,1:].values)) #0-255숫자를 배분해서 넣어둠
# test_data=list(map(func,test_csv.iloc[:,1:].values))


# iloc : 행으로 가져옴
# train_data2=np.array(list(train.iloc[:,1:].values))
# 정규화, 표준화작업
train_data=train.iloc[:,1:].values
test_data=test.iloc[:,1:].values
train_scaled=train_data/255.0
test_scaled=test_data/255.0

train_label=train[0].values
test_label=test[0].values


# # 알고리즘선택 - 다차원배열 사용가능 (머신러닝: (784,), 딥러닝 : (28,28))
model=keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(784,))) # input의 개수 맞추기
model.add(keras.layers.Dense(100,activation="sigmoid"))
model.add(keras.layers.Dense(10,activation="softmax")) # 2개면 sigmoid 아니라면 softmax
model.compile(loss='sparse_categorical_crossentropy',metrics="accuracy") # sparse을 사용하게 되면 알아서 원핫인코딩으로 변환됨

# 훈련
model.fit(train_scaled,train_label,epochs=5)

# 딥러닝 model요약문
# 784*100+100=78500, 100*10+10=1010
print(model.summary()) # param : 계산한값, 가중치?

# 정확도
score=model.evaluate(test_scaled,test_label)
print(score)