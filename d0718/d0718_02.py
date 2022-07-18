# 형태소 분석
from konlpy.tag import Okt
# 필요한 단어만 추출해서 사용하기 => 전처리과정이라고 볼수 있음

okt=Okt()
text='한글 자연어 처리는 재밌다. 이제부터 열심히 해야지 ㅎㅎㅎ'

# 텍스트 단위로 형태소 분리
print(okt.morphs(text))

# 명사만 추출
print(okt.nouns(text))

# 어절단위로 추출
print(okt.phrases(text))

# 품사도 함께 추출 (튜플형태)
print(okt.pos(text))
# 품사와 함께 추출 '/' 형태로 나옴
print(okt.pos(text,join=True))

