# def numbers(): #계산기
#     X=[]
#     while True:
#         number = input("Enter a number (<Enter key> to quit)")
#         while number !="":
#             try:
#                 x = float(number)
#                 X.append(x)
#             except ValueError:
#                 print('>>> NOT a number! Ignored..')
#             number = input("Enter a number (<Enter key> to quit)")
#         if len(X) > 1:
#             return X
#
# def median(nums):
#     nums.sort()
#     size = len(nums)
#     p = size // 2
#     if size % 2 == 0:
#         pr = p
#         pl = p-1
#         mid = float((nums[pl]+nums[pr])/2)
#     else:
#         mid = nums[p]
#     return mid
#
# def means(nums):
#     total = 0.0
#     for i in range(len(nums)):
#         total = total + nums[i]
#     return total / len(nums)
#
# def std_dev(nums, avg):
#    texp = 0.0
#    for i in range(len(nums)):
#        texp = texp + (nums[i] - avg) ** 2
#    return (texp/(len(nums)-1)) ** 0.5
#
# def main():
#     X = numbers()
#     med = median(X)
#     avg = means(X)
#     std = std_dev(X, avg)
#     print("당신이 입력한 숫자{}의 ".format(X))
#     print("중앙값은{}, 평균은{}, 표준편차는{}입니다.".format(med, avg, std))
#
# if __name__ == '__main__':
#     main()
# #
#
# treasure_box = {'rope': {'coin': 1, 'pcs': 2},
#                 'apple': {'coin': 2, 'pcs': 10},
#                 'torch': {'coin': 2, 'pcs': 6},
#                 'gold coin': {'coin': 5, 'pcs': 50},
#                 'knife': {'coin': 30, 'pcs': 1},
#                	'arrow': {'coin': 1, 'pcs': 30}
#                }
# coin_per_treasure = {'rope':1,
#         'apple':2,
#         'torch': 2,
#         'gold coin': 5,
#         'knife': 30,
#         'arrow': 1}
#
#
# def display_stuff(treasure_box):
#     ## type your code
#     print("Congraturation!! you got a treasure box!!")
#     for treasure in treasure_box:
#              print("You have {} {}pcs".format(treasure, treasure_box[treasure]['pcs']))
# display_stuff(treasure_box)
#
# def total_silver(treasure_box, coin_per_treasure):
#     ## type your code
#     total_coin = 0
#     for treasure in treasure_box:
#         coin = coin_per_treasure[treasure] * treasure_box[treasure]['pcs']
#         print("{} : {}coins/pcs * {}pcs = {} coins".format(
#           treasure, coin_per_treasure[treasure], treasure_box[treasure]['pcs'], coin))
#         total_coin += coin
#     print('total_coin : ', total_coin)
# total_silver(treasure_box, coin_per_treasure)
