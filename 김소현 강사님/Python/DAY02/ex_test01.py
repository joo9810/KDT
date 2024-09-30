#-------------------------------------------
# 입력 & 출력 실습
#-------------------------------------------
# [실습1] 데이터 저장 및 변수 생성 그리고 출력
# - 생년월일
# - 띠 (용,범...)
# - 혈액형
# - 출력형태
#   나는 0000년 00월 00일 000띠입니다
#   혈액형은 ____ 0형입니다
year='1998'
month='10'
day='01'
c_zodiac='호랑이'
feature='다재다능'
blood='O'

print(f'나는 {year}년 {month}월 {day}일 {c_zodiac}띠입니다')
print(f'혈액형은 {feature}한 {blood}형입니다')
#-------------------------------------------


# [실습2] 입력 받은 데이터 저장 단, 파일로 저장------
# - 좋아하는 계절
# - 좋아하는 나라
# - 여행가고 싶은 나라
fav_season=input('좋아하는 계절을 입력하세요')
fav_con=input('좋아하는 나라를 입력하세요')
travel_con=input('여행가고 싶은 나라를 입력하세요')

FILENAME='test2.txt'
f=open(FILENAME, mode='w')
# f.write(fav_season)
# f.write(fav_con)
# f.write(travel_con)
print(f'좋아하는 계절 : {fav_season}', file=f)
print(f'좋아하는 나라 : {fav_con}', file=f)
print(f'여행가고 싶은 나라 : {travel_con}', file=f, end='')
# print(f'{fav_season}\n{fav_con}\n{travel_con}\n', file=f)
f.close()
#-------------------------------------------