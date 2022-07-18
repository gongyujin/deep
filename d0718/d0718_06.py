# 워드클라우드형태 출력
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text='파이썬 파이썬 파이썬 워드클라우드 워드클라우드 라이브러리 좋아 좋아 예시 워드클라우드 워드클라우드 데이터분석 데이터 분석\
    파이썬 파이썬 파이썬 파이썬 딥러닝 딥러닝 딥러닝 머신러닝 집 가고 싶다 집집집집집집집집'

# 한글적용해주려면 해야함
wordcloud=WordCloud('deep/d0718/MALGUN.TTF').generate(text)
# interpolation='bilinear' : 글자 부드럽게
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# 0 : 글이 없는것
# 2 : 글은 있는데 포함되지 않는것
# target : 0 - 부정, 1 - 긍정
# simpleRNN은 원핫인코딩 해줘야함 => 임베딩하면 안해줘도 됨


