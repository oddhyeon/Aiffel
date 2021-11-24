import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_wine
#
# data = load_wine()
# type(data)
#
# r = np.random.RandomState(10)
# x = 10 * r.rand(100)
# y = 2 * x - 3 * r.rand(100)
# # plt.scatter(x,y)
# # plt.show()
# # print(x.shape,y.shape)
# model = LinearRegression()
# X = x.reshape(100,1)
# model.fit(X,y)
#
# x_new = np.linspace(-1, 11, 100)
# X_new = x_new.reshape(100,1)
# y_new = model.predict(X_new)
#
# plt.scatter(x, y, label='input data')
# plt.plot(X_new, y_new, color='red', label='regression line')

# print(data.keys())
# print(data.data)    # 키에 특성
# print(data.data.shape) # 데이터 갯수, 특성 갯수
# print(data.data.ndim)   # 몇 차원인지
# print(data.target)  # 타겟 벡터
# print(data.target.shape) # 특성 행렬 데이터 수와 일치
# print(data.feature_names) # 13개의 feature_names란 키에 특성의 이름
# print(len(data.feature_names))    위 특성의 갯수
# print(data.target_names)    # 분류하고자 하는 대상
# print(data.DESCR) # 약자로 데이터에 대한 설명

# import pandas as pd
# print(pd.DataFrame(data.data, columns=data.feature_names)) # 변수명 X에 저장하고, 타겟 벡터는 y에 저장
# X = data.data
# y = data.target
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()    # 모델 생성
# model.fit(X, y)     # 모델 훈련
# y_pred = model.predict(X)   # 모델 예측
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
#
# #타겟 벡터 즉 라벨인 변수명 y와 예측값 y_pred을 각각 인자로 넣습니다. 분류문제에 경우 아래의 성능 평가 툴 사용
# print(classification_report(y, y_pred))
# #정확도를 출력합니다.
# print("accuracy = ", accuracy_score(y, y_pred))
#
# from sklearn.datasets import load_wine
# data = load_wine()
# print(data.data.shape)
# print(data.target.shape)


# from sklearn.datasets import load_wine
# data = load_wine()
# print(data.data.shape)
# print(data.target.shape) # 178개의 데이터 확인
#
# X_train = data.data[:142] # 슬라이싱으로 트레이닝셋과 테스트셋을 분리
# X_test = data.data[142:]
# print(X_train.shape, X_test.shape)
#
# y_train = data.target[:142] # 슬라이싱으로 트레이닝셋과 테스트셋을 분리
# y_test = data.target[142:]
# print(y_train.shape, y_test.shape)
#
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train) # 모델 훈련
# y_pred = model.predict(X_test)    #모델 예측
# from sklearn.metrics import accuracy_score
#
# print("정답률=", accuracy_score(y_test, y_pred)) # 정확도 평가



# from sklearn.model_selection import train_test_split
#
# result = train_test_split(X, y, test_size=0.2, random_state=42)
# print(type(result))
# print(len(result))
# print(result[0].shape , result[1].shape, result[2].shape, result[3].shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# from sklearn.datasets import load_iris  # iris(붓꽃 품종 종류)
# data =load_iris()
# print(type(data))
# print(data)
# print(data.keys())
# print(data.data.shape)
# print(data.data.ndim)
# print(data.target.shape)
# print(data.feature_names)
# print(data.target_names)
# # print(data.DESCR) # 데이터 설명
# X = data.data
# y = data.target
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# # print(data.data.shape)
# # print(data.target.shape)
# X_train = data.data[:120]
# X_test = data.data[120:]
# # print(X_train.shape, X_test.shape)
# y_train = data.target[:120]
# y_test = data.target[120:]
# # print(y_train.shape, y_test.shape)
#
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# from sklearn.metrics import accuracy_score
#
# print("정답률=", accuracy_score(y_test, y_pred))
#
# from sklearn.model_selection import train_test_split
#
# result = train_test_split(X, y, test_size=0.2, random_state=42)
# print(type(result))
# print(len(result))


from sklearn.datasets import load_iris  # iris(붓꽃 품종 종류)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터셋 로드하기
data = load_iris() # 데이터 로드
# 훈련용 데이터셋 나누기
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=30) # 시드 고정
# 훈련하기
model = RandomForestClassifier() # 모델 정의
model.fit(X_train, y_train) # 모델 훈련
# 예측하기
y_pred = model.predict(X_test)  # 모델 예측
# 정답률 출력하기
print("정답률=", accuracy_score(y_test, y_pred))

