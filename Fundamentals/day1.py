# 실용 파이썬 프로그래밍 1.1~1.4
# ball = 100
# count = 0
# while count < 10:
#     count = count + 1
#     ball = ball * 0.6
#     print(count, f"{ball:0.4f}")
#
# # sears.py
# """어느 날 아침, 당신은 시카고의 시어스 타워(Sears tower) 근처를 거닐다가 보도에 1 달러 지폐를 한 장 올려뒀다.
# 그 후 매일 외출할 때마다 그 위에 지폐를 얹어 탑을 쌓으며, 높이는 매일 두 배로 불어난다.
# 돈으로 쌓은 탑의 높이가 시어스 타워의 높이와 같아지려면 시간이 얼마나 걸릴까?"""
# bill_thickness = 0.11 * 0.001    # 미터(0.11 mm)
# sears_height   = 442             # 높이(미터)
# num_bills      = 1
# day            = 1
#
# while num_bills * bill_thickness < sears_height:
#     print(day, num_bills, num_bills * bill_thickness)
#     day = day + 1
#     num_bills = num_bills * 2
#
# print('Number of days', day)
# print('Number of bills', num_bills)
# print('Final height', num_bills * bill_thickness)


# mortgage.py 1.3 문제
""" 데이브는 500,000 달러의 30년 고정 이율 주택 담보 대출(mortgage)을 받기로 결정했다. 
이율은 5%이고 매달 납부할 금액은 2684.11 달러다.
다음은 대출 기간 동안 지불할 총액을 계산하는 프로그램이다. """
# principal = 500000.0
# rate = 0.05
# payment = 0
# total_paid = 0.0
# additional_entry = 0
# total_count = 0
# extra_payment_start_month = 61
# extra_payment_end_month = 108
# extra_payment = 1000
#
# while principal > payment:
#     principal = principal * (1+rate/12) - payment
#     if extra_payment_start_month <= total_count and total_count <= extra_payment_end_month:
#         payment = 3684.11
#         total_paid = total_paid + payment
#     else:
#         payment = 2684.11
#         total_paid = total_paid + payment
#     total_count = total_count + 1
#     print(f'{total_count} {total_paid:0.2f} {principal:0.2f}')
#
# print(f'Total paid {total_paid:0.2f} \nMonths {total_count}')
#
#
#
print("1")