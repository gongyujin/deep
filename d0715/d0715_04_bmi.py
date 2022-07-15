from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
bmi=pd.read_csv('deep/d0715/bmi.csv')

# ######## 원핫인코딩 컬럼추가#########################
bmi_label = pd.get_dummies(bmi['label'])
bmi=bmi.drop('label',axis=1)
bmi=bmi.join(bmi_label)

# 2. 데이터 전처리
# # 정규화, 표준화 작업
# 정규화
bmi['height']=bmi['height']/200
bmi['weight']=bmi['weight']/100

data=bmi[['height','weight']]
label=bmi[['fat','normal','thin']].to_numpy()

# # 3. label -> 원핫인코딩으로 변경
# # class : 3개 thin, normal, fat
# label_class={'thin':[1,0,0],'normal':[0,1,0],'fat':[0,0,1]}
# y_label=np.zeros((20000,3))
# print(y_label)
# for i,v in enumerate(bmi['label']):
#     # i=0, v=normal
#     y_label[i]=label_class[v]
# print(y_label)

# train, test 데이터 분리
train_scaled,val_scaled,train_label,val_label=train_test_split(data,label)

# ---------------------------------------------
# 4. 딥러닝 선언
# # 같은 의미
# model.add(keras.layers.Dense(100,activation='relu')) 
# #-------------------------------------------------------
# model.add(keras.layers.Dense(100))
# model.add(activation='relu')

model=keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(2,)))
model.add(keras.layers.Dense(100,activation='relu'))
# model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

# 조기종료
early_stop=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)

# 5. 딥러닝 훈련
history=model.fit(train_scaled,train_label,epochs=20,\
    # batch_size=128, 
    validation_data=(val_scaled,val_label),callbacks=[early_stop])
print(model.summary())

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.xlabel('loss')
# plt.legend(['train','val'])
# plt.show()

# 6. 정확도
score=model.evaluate(val_scaled,val_label)
print('정확도 : ', score)

# # 7. (141,64) 예측(분류) 하시오

predict=np.argmax(model.predict([[161/200,28/100]]),axis=1)
# predict=np.argmax(model.predict([[141/200,64/100]]),axis=1)
# predict2=np.argmax(model.predict([[0.91,0.79]]),axis=1)
# predict3=np.argmax(model.predict([[0.68,0.63]]),axis=1)
# predict4=np.argmax(model.predict([[0.82,0.75]]),axis=1)

print('result : ', predict) 
# print('result : ', predict2) 
# print('result : ', predict3) 
# print('result : ', predict4) 