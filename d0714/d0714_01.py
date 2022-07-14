from tensorflow import keras
import sklearn
from sklearn.linear_model import SGDClassifier # 확률적경사하강법
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# mnist파일 불러오기
# keras mnist파일이 설치될때 자동설치가 됨
# 튜플형태로 자동적으로 가져오게 됨
(train_data,train_label),(test_data,test_label)=keras.datasets.fashion_mnist.load_data()
print(train_data.shape, train_label.shape) # 28*28 이 6만개
print(test_data.shape,test_label.shape)

# print(train_label[:5]) # 0~9까지 이미지를 의미
# 이미지 출력
print(train_label[:10])
fig,axs=plt.subplots(1,10,figsize=(10,10))
for i in range(10):
    # imshow: 그레이출력, gray_r : 반전
    axs[i].imshow(train_data[i],cmap='gray_r')
    axs[i].axis('off')
plt.show()

# 전처리
# 60000,28,28에서 60000, 28*28 형태로 변환해줘야함 => 즉, 3차원을 2차원으로 전처리시켜줘야함


# 알고리즘 선택

# 실습훈련

# 정확도