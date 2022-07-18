from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd
from gensim.models import Word2Vec

# 토지의 책 불러오기
f=open('deep/d0718/BEXX0003.txt','r',encoding='utf-16')
soup=BeautifulSoup(f,'html.parser')
body=soup.select_one('body > text').get_text()

# 형태소분석 각각의 단어 몇번씩 나오는지 출력
okt=Okt()
lines=body.split('\n')

word_dic={}
for line in lines:
    malist=okt.pos(line,norm=True,stem=True)
    for taeso, pumsa in malist:
        if pumsa=='Noun':
            if not (taeso in word_dic):
                word_dic[taeso]=0
            word_dic[taeso]+=1

keys=sorted(word_dic.items(), key=lambda x: x[1],reverse=True)
for word, count in keys[:]:
    print('{}:{}'.format(word,count),end=' ')
    
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# Word2Vec 벡터화
token_data=[]
for line in lines:
    temp_x=okt.morphs(line,stem=True)
    temp_x=[word for word in temp_x if not word in stopwords]
    token_data.append(temp_x)

# 김서방 : 연관글을 찾아보시오.
model=Word2Vec(sentences=token_data,vector_size=100,window=5,min_count=3,workers=4,sg=0)
print(model.wv.most_similar(positive=['김서방']))