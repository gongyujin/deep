from konlpy.tag import Okt

okt=Okt()

# pos : 형태소분석, norm : 그래욬ㅋㅋ -> '그래요'로 변환시켜서 저장시켜줌, stem : 잘나가는 -> 잘나가다
print(okt.pos('이것도 되낰ㅋㅋㅋㅋㅋㅋ',norm=True,stem=True))

# print(okt.morphs('단독 입찰보다 복수 입찰일 경우')) # 형태소 모두 추출 : morphs
# print(okt.nouns('유일하게 항공기 제작이 가능한 곳입니다.')) # 명사만 추출 : nouns
# print(okt.pos('유일하게 항공기 제작이 가능한 곳입니다.')) # 글, 품사 추출 : pos 
# print(okt.phrases('날카로운 분석과 신뢰감 있는 진행으로')) # 텍스트 어절 추출 : phrase  

