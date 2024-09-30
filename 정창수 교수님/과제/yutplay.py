## 윳놀이 게임 프로그램
'''
흥부와 놀부가 윷놀이를 하는데,
각자 4개의 윷가락을 교대로 던져서 20점 이상의 점수가 먼저 나오는 사람이 승리를 한다.
윷가락을 던져 나온 점수가 "윷(4점)"이나 "모(5점)"가 나오는 경우,
동일한 사람이 한 번 더 던질 수 있다.
아래 주어진 메소드를 구현하고 필요한 기능들은 추가로 구현하여 프로그램을 작성하시오.
'''

import random

def generate_mo():
    sticks = [1, 1, 1, 1]
    return sticks

def generate_yut():
    sticks = []
    for _ in range(4):
        sticks.append(random.randint(0, 1))

    return sticks

hungbu_score = []
nolbu_score = []

order = random.randint(0, 1)

if order == 0: # 0이면 흥부 먼저
    current_turn = hungbu_score
else:          # 1이면 놀부 먼저
    current_turn = nolbu_score

while True:
    current_yut = generate_yut() # 윳 생성

    # current_yut = generate_mo() # 강제로 모만 나오게 하기

    if current_yut == [1, 1, 1, 1]:
        score = 5
        name = '모'
        current_turn.append(score)

    elif current_yut == [0, 0, 0, 0]:
        score = 4
        name = '윳'
        current_turn.append(score)

    elif sum(current_yut) == 1:
        score  = 3
        name = '걸'
        current_turn.append(score)

    elif sum(current_yut) == 2:
        score = 2
        name = '개'
        current_turn.append(score)

    else:
        score = 1
        name = '도'
        current_turn.append(score)

    if current_turn == hungbu_score:
        print(f'흥부 {current_yut}: {name} ({score}점)/(총 {sum(hungbu_score)}점) --->')
        if score == 4 or score == 5:
            current_turn == hungbu_score
        else:
            current_turn = nolbu_score
    else:
        print(f'{" " * 33}<--- 놀부 {current_yut}: {name} ({score}점)/(총 {sum(nolbu_score)}점)')
        if score == 4 or score == 5:
            current_turn == nolbu_score
        else:
            current_turn = hungbu_score
    
    if sum(hungbu_score) >= 20 or sum(nolbu_score) >= 20:
        break

print('-' * 75)
if sum(hungbu_score) > sum(nolbu_score):
    print(f'흥부 승리 => 흥부: {sum(hungbu_score)} , 놀부: {sum(nolbu_score)}')
else:
    print(f'놀부 승리 => 흥부: {sum(hungbu_score)} , 놀부: {sum(nolbu_score)}')

print('-' * 75)