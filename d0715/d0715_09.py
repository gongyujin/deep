from konlpy.tag import Okt
import pandas as pd
from bs4 import BeautifulSoup
import codecs

fp=codecs.open('deep/d0715/BEXX0003.txt','r',encoding='utf-16')
soup=BeautifulSoup(fp,'html.parser')
body=soup.select_one('body > text').get_text()

# ------------------------------------------------------------
# 1글자 제외 명사만 출력 해보세요.
okt=Okt()
word_dic=[]
lines=body.split('\n') # 3440개
for line in lines:
    malist=okt.pos(line,norm=True,stem=True)
    r=[]
    for taeso, pumsa in malist:
        if pumsa=='Noun':
            if len(taeso)>=2:
                if not (taeso in word_dic):
                    word_dic[taeso]=0
                word_dic[taeso]+=1
# ------------------------------------------------------------

keys=sorted(word_dic.items(), key=lambda x: x[1],reverse=True)
for word, count in keys[:]:
    print('{}:{}'.format(word,count))