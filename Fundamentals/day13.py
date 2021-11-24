from sklearn.datasets import load_iris
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

print(dir(iris))
dir()#는 객체가 어떤 변수와 메서드를 가지고 있는지 나열함

print(iris.keys())
print(iris_data.shape)
#shape는 배열의 형상정보를 출력
print(iris_data[0])
print(iris_label.shape)
print(iris_label)
print(iris.target_names)

#import pandas as pd
# iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names) # 데이터 프레임으로 자료형 변환
# # print(iris_df)
# iris_df["label"] = iris.target # 타겟 컬럼 추가
# # print(iris_df)
#
# from sklearn.model_selection import train_test_split  # 학습 데이터와,테스트 데이터 분리 및 시드 고정
#
# X_train, X_test, y_train, y_test = train_test_split(iris_data,
#                                                     iris_label,
#                                                     test_size=0.2,
#                                                     random_state=7)
#
# # print('X_train 개수: ', len(X_train), ', X_test 개수: ', len(X_test))
# # print(X_train.shape, y_train.shape)
# # print(X_test.shape, y_test.shape)
# # print(y_train, y_test)
#
# from sklearn.tree import DecisionTreeClassifier
#
# decision_tree = DecisionTreeClassifier(random_state=32) # 모델 정의
# decision_tree.fit(X_train, y_train) # 모델 학습
# y_pred = decision_tree.predict(X_test) # 모델 예측값
# print(y_pred,'\n')
#
# print(y_test) # 실제 정답
# from sklearn.metrics import accuracy_score
#
# accuracy = accuracy_score(y_test, y_pred) # 정확도 측정
# print(accuracy)

# #위 코드를 아래 짧게 요약
# # (1) 필요한 모듈 import
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report
#
# # (2) 데이터 준비
# iris = load_iris()
# iris_data = iris.data
# iris_label = iris.target
#
# # (3) train, test 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(iris_data,
#                                                     iris_label,
#                                                     test_size=0.2,
#                                                     random_state=7)
#
# # (4) 모델 학습 및 예측
# decision_tree = DecisionTreeClassifier(random_state=32)
# decision_tree.fit(X_train, y_train)
# y_pred = decision_tree.predict(X_test)
#
# print(classification_report(y_test, y_pred))
#
#
#
# from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier 모델 정의
#
# X_train, X_test, y_train, y_test = train_test_split(iris_data,
#                                                     iris_label,
#                                                     test_size=0.2,
#                                                     random_state=21)
#
# random_forest = RandomForestClassifier(random_state=32)
# random_forest.fit(X_train, y_train)
# y_pred = random_forest.predict(X_test)
#
# print(classification_report(y_test, y_pred))
#
#
# from sklearn import svm   # SVM 모델 정의 및 실행
# svm_model = svm.SVC()
#
# svm_model.fit(X_train, y_train)
# y_pred = svm_model.predict(X_test)
#
# print(classification_report(y_test, y_pred))
#
# from sklearn.linear_model import SGDClassifier # SGDClassifier 모델 정의
# sgd_model = SGDClassifier()
#
# sgd_model.fit(X_train, y_train)
# y_pred = sgd_model.predict(X_test)
#
# print(classification_report(y_test, y_pred))
#
#
# from sklearn.linear_model import LogisticRegression # Logistic Regression 모델 정의
# logistic_model = LogisticRegression()
#
# logistic_model.fit(X_train, y_train)
# y_pred = logistic_model.predict(X_test)
#
# print(classification_report(y_test, y_pred))



from sklearn.datasets import load_digits  # 손글씨 데이터 MNIST 데이터셋

digits = load_digits()
digits.keys()
digits_data = digits.data
# print(digits_data.shape)  # (1797, 64) 데이터
# print(digits_data[0])  # 이미지라 8*8 숫자 배열 출력


import matplotlib.pyplot as plt # 이미지 출력시 matplotlib 활용

plt.imshow(digits.data[0].reshape(8, 8), cmap='gray')
plt.axis('off')
# plt.show()


for i in range(10):  # 0~9까지 이미지 확인
    plt.subplot(2, 5, i+1)   # 행,열 및 이미지순차
    plt.imshow(digits.data[i].reshape(8, 8), cmap='gray')
    plt.axis('off')  # x,y축 범위 설정 함수 (좌측 코드는 축과 라벨을 삭제한다)
# plt.show()

digits_label = digits.target
# print(digits_label.shape)
digits_label[:20]  # 20까지의 정답 데이터
new_label = [3 if i == 3 else 0 for i in digits_label]
new_label[:20]  # 위 조건을 사용하여 정답일경우 그대로 3 출력 나머지는 0을 출력

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                    new_label,
                                                    test_size=0.2,
                                                    random_state=15)

decision_tree = DecisionTreeClassifier(random_state=15)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
# print(accuracy)

fake_pred = [0] * len(y_pred)  # 0으로만 이루어진 리스트 생성

accuracy = accuracy_score(y_test, fake_pred)  # y_test간 정확도 확인
# print(accuracy)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))