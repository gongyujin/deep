from konlpy.tag import Okt
import pandas as pd

# book 파일 가져오기
f= open('deep/d0715/book.txt',encoding='utf-8')
book = f.read()
# print(book)

# 행태소분석
okt=Okt()

# {'날카로운':1,'분석':10}
word_dic={}
lines=book.split('\n') # 1402개
# print(lines[3])

# ------------------------------------------------------
for line in lines:
    # 형태소 변환
    # 명사인것만 불러오기
    malist=okt.pos(line,norm=True,stem=True)
    for taeso, pumsa in malist:
        # 해당 글내용, 품사
        # if pumsa in ['Noun','Josa','Verb']:
        if pumsa == 'Noun':
            # word_dic안에 형태소가 있는지 확인
            if not (taeso in word_dic):
                word_dic[taeso]=0 # 자리만들어주기
            word_dic[taeso] +=1
    
#------------------------------------------------------
# 숫자 역순정렬
keys=sorted(word_dic.items(),key=lambda x :x[1],reverse=True)

# 튜플형태의 리스트
# 50개 출력
for word, count in keys[:50]:
    print('{}:{}'.format(word,count),end=' ')
    