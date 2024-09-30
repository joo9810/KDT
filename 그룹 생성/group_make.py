import random


# 그룹 생성 함수
def generate_group():
    # 학생 리스트
    student = ['박란영', '조혜리', '전민규', '안효준', '김경태', '김태헌', '구성윤', '황은혁',
            '김경환', '곽경민', '김재성', '도영훈', '한세진', '이민하', '장재웅', '이현종',
            '박형준', '손지원', '황지원', '김환선', '김민석', '이동건', '김영주', '안윤호',
            '김미소', '김아란', '김도연', '권도운', '김이현', '김현주', '이송은', '박지훈']

    group_dict = {}

    for i in range(8):
        group_list = []
        for j in range(4):
            idx = random.choice(range(32-(i*4+j))) # 학생 한 명을 뽑을때마다 리스트 길이가 1씩 줄어듦
            group_list.append(student.pop(idx)) # pop을 사용해서 비복원 추출
        group_dict[f'{i+1}조'] = group_list # 딕셔너리에 저장

    # team.txt 파일로 조 정보 저장
    with open('current_teams.txt', 'w', encoding='utf-8') as file:
        for group_number, group_member in group_dict.items():
            file.write(f'{group_number}: {", ".join(group_member)}\n')

    return group_dict

# 이전 프로젝트 그룹과 중복이 없는지 검사하는 함수
def valid_check(first_group, second_group):
    for i in range(8):
        for j in range(8):
            # 만약 첫 번째 그룹과 두 번째 그룹의 교집합이 2이상인 곳이 존재하면 False 반환
            if len(set(list(first_group.values())[i]) & set(list(second_group.values())[j])) >= 2:
                return False
    return True


# 중복이 없게 조 생성
def without_duplicates(all_group):
    while True:
        is_valid = True # 플래그 변수 선언
        if len(all_group) == 0: # 첫 프로젝트 생성
            all_group.append(generate_group())
            break
        elif len(all_group) >= 3: # 진행한 프로젝트가 3개 이상이 될때
            all_group = all_group[len(all_group)-2:len(all_group)]
            all_group.append(generate_group()) # all_group에 생성한 조 추가하기
            for i in range(len(all_group)-1):
                for j in range(i+1, len(all_group)):
                    if valid_check(all_group[i], all_group[j]) == False:
                        is_valid = False
                        all_group.pop()
                        break
                if is_valid == False:
                    break
            if is_valid == True:
                break
        else: # 진행한 프로젝트가 2개일때
            all_group.append(generate_group()) # all_group에 생성한 조 추가하기
            for i in range(len(all_group)-1):
                for j in range(i+1, len(all_group)):
                    if valid_check(all_group[i], all_group[j]) == False:
                        is_valid = False
                        all_group.pop()
                        break
                if is_valid == False:
                    break
            if is_valid == True:
                break
        
    return all_group