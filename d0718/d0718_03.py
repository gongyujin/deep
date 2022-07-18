from konlpy.tag import Okt
from lightgbm import train
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request
# from gensim.models import word2vec
import gensim

# url 파일 불러오기
# 영화평점
urllib.request.urlretrieve('https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt',filename='ratings.txt')
train_data=pd.read_table('ratings.txt')

## 데이터 확인
# print(train_data.describe()) # document형태는 나오지 않음
# print(train_data.info())
# print(train_data['label']) # 부정적인 글, 긍정적인 글

## 총개수 - 200000
# print(len(train_data))

# null값 제거 - 199992
train_data=train_data.dropna(how='any')
# print(train_data.info())

# # 한글 외 모든 글자 제외 nnn/n
# # 영화 댓글 부분
# # regex=True : 문자열 일부 치환설정
# print(train_data.head())
train_data['document']=train_data['document'].str.replace('[^ㄱ-하-ㅣ가-힣]','',regex=True)
# print(train_data.head())

# 형태소 분석
okt=Okt()
# word2vec 형태소 분석 -> 글자간의 벡터화를 해서 글자간의 관계를 형성
# [[]] 2차원이어야함

## 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


token_data=[] # 제일 바깥 리스트
# 199992개를 가져와서 형태소 분석
for sent in train_data['document']:
    # 1줄을 가져와서 형태소 분석
    temp_x = okt.morphs(sent,stem=True) # 형태소 만들어져서 temp_x에 저장
    temp_x=[word for word in temp_x if not word in stopwords]
    token_data.append(temp_x)
    
# word2vec - 글과 글의 관계가 형성됨
# min_count : 5개 이하는 번호를 매기지않음(찾지않음..)
model = gensim.models.word2vec(sentences=token_data,vector_size=100,window=5,min_count=5,workers=4,sg=0)

# # 모델 save
# model.save('model_w2v.h5')
# model=word2vec.load('model_w2v')

print(model.wv.most_similar('때'))