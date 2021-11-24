# # 다음 소스 코드에서 동물 클래스 Animal과 조류 클래스 Wing을 상속받아 새 클래스
# # Bird를 작성하여 '먹다', '파닥거리다', '날다'
# # 파충류 ,양서류 ,어류 ,조류
#
# class Animal:
#
#     def __init__(self,breathe,eat):
#         self.breathe = breathe
#         self.eat = eat
#
#
# class Wing(Animal):
#
#     def __init__(self,breathe,eat,fly):
#         super().__init__(breathe,eat)
#         self.fly = fly
#
# lion = Animal('air','meet')
# sparrow = Wing('air','banban','True')
#
# print(lion.eat)
# print(sparrow.fly)

# import time
#
# start = time.time()  # 시작 시간 저장
#
# a = 1
# for i in range(100):
#     a += 1
#
# # 작업 코드
#
# my_list = ['a','b','c','d']
#
# for i in my_list:
#     print("값 : ", i)
#
# print("time :", time.time() - start)  # 결과는 '초' 단위 입니다.
#
# my_list = ['a','b','c','d']
#
# for i, value in enumerate(my_list):
#     print("순번 : ", i, " , 값 : ", value)
#
# my_list = ['a', 'b', 'c', 'd']
# result_list = []
#
# for i in range(2):
#     for j in my_list:
#         result_list.append((i, j))
#
# print(result_list)
#
# result_list = [(i, j) for i in range(2) for j in my_list]
#
# print(result_list)
#
# my_list = ['a','b','c','d']
#
# # 인자로 받은 리스트로부터 데이터를 하나씩 가져오는 제너레이터를 리턴하는 함수
# def get_dataset_generator(my_list):
#     result_list = []
#     for i in range(2):
#         for j in my_list:
#             yield (i, j)   # 이 줄이 이전의 append 코드를 대체했습니다
#             print('>>  1 data loaded..')
#
# dataset_generator = get_dataset_generator(my_list)
# for X, y in dataset_generator:
#     print(X, y)
#
import array as arr

mylist = [1, 2, 3]   # 이것은 파이썬 built-in list입니다.
print(type(mylist))

mylist.append('4')  # mylist의 끝에 character '4'를 추가합니다.
print(mylist)

mylist.insert(1, 5)  # mylist의 두번째 자리에 5를 끼워넣습니다.
print(mylist)

myarray = arr.array('i', [1, 2, 3])   # 이것은 array입니다. import array를 해야 쓸 수 있습니다.
print(type(myarray))

# 아래 라인의 주석을 풀고 실행하면 에러가 납니다.
#myarray.append('4')    # myarray의 끝에 character '4'를 추가합니다.
print(myarray)

myarray.insert(1, 5)    # myarray의 두번째 자리에 5를 끼워넣습니다.
print(myarray)

med = median(X)
avg = means(X)
std = std_dev(X, avg)
print("당신이 입력한 숫자{}의 ".format(X))
print("중앙값은{}, 평균은{}, 표준편차는{}입니다.".format(med, avg, std))