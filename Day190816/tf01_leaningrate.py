#1. 데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 모델링 과정
model.add(Dense(5, input_dim = 1, activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(3))
model.add(Dense(4))

model.add(Dense(1))

#3. 훈련
from keras.optimizers import Adam
optimizer = Adam(lr=0.00996)
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)       # 훈련(실행)시키다

#4. 평가 예측
mse, _ = model.evaluate(x, y, batch_size=1)
print('mse : ', mse)
pred1 = model.predict([1.5,2.5,3.5])
print(pred1)