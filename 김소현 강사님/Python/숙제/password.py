import random

# 크기가 15인 리스트 생성
list_data = [0] * 15

# 가능한 숫자 리스트
num_list = list(range(10))

# 가능한 영어 소문자 리스트 
lower_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]

# 가능한 영어 대문자 리스트
upper_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# 가능한 특수문자 리스트 (!, @, #, $, %, ^, &로 제한)
special_list = ['!', '@', '#', '$', '%', '^', '&']


# 영문자 + 숫자 조합 생성기
def generator1(seed_number):
    random.seed(random.random() + seed_number)

    # 15자리의 비밀번호 자리 중 각 자리에 뭐가 들어올지 정하기
    # 랜덤으로 뽑은 숫자가 0: 숫자, 1: 영어 소문자, 2: 영어 대문자
    select_list = [random.randint(0, 2) for _ in range(15)]


    for i in range(15):
        # 만약 select_list[i]가 숫자라면
        if select_list[i] == 0:
            # 숫자가 들어있는 리스트의 인덱스 랜덤 추출
            idx1 = random.randint(0, len(num_list)-1)
            list_data[i] = num_list[idx1]

        # 만약 select_list[i]가 영어 소문자라면
        elif select_list[i] == 1:
            # 영어 소문자가 들어있는 리스트의 인덱스 랜덤 추출
            idx2 = random.randint(0, len(lower_list)-1)
            list_data[i] = lower_list[idx2]

        # 만약 select_list[i]가 영어 대문자라면
        else:
            # 영어 소문자가 들어있는 리스트의 인덱스 랜덤 추출
            idx3 = random.randint(0, len(upper_list)-1)
            list_data[i] = upper_list[idx3]


    return ''.join(list(map(str, list_data)))


# 특수문자 + 영문자 + 숫자 조합 생성기
def generator2(seed_number):
    random.seed(random.random() + seed_number)


    # 15자리의 비밀번호 자리 중 각 자리에 뭐가 들어올지 정하기
    # 랜덤으로 뽑은 숫자가 0: 숫자, 1: 영어 소문자, 2: 영어 대문자, 3: 특수문자
    select_list = []
    for _ in range(15):
        select_list.append(random.randint(0, 3))


    for i in range(15):
        # 만약 select_list[i]가 숫자라면
        if select_list[i] == 0:
            # 숫자가 들어있는 리스트의 인덱스 랜덤 추출
            idx1 = random.randint(0, len(num_list)-1)
            list_data[i] = num_list[idx1]

        # 만약 select_list[i]가 영어 소문자라면
        elif select_list[i] == 1:
            # 영어 소문자가 들어있는 리스트의 인덱스 랜덤 추출
            idx2 = random.randint(0, len(lower_list)-1)
            list_data[i] = lower_list[idx2]

        # 만약 select_list[i]가 영어 대문자라면
        elif select_list[i] == 2:
            # 영어 소문자가 들어있는 리스트의 인덱스 랜덤 추출
            idx3 = random.randint(0, len(upper_list)-1)
            list_data[i] = upper_list[idx3]

        # 만약 select_list[i]가 특수문자라면   
        else:
            # 특수문자가 들어있는 리스트의 인덱스 랜덤 추출
            idx4 = random.randint(0, len(special_list)-1)
            list_data[i] = special_list[idx4]



    return ''.join(list(map(str, list_data)))


def print_menu1():
    print(f'{"*":*^24}')
    print(f'*{"비밀번호 생성기":^15}*')
    print(f'{"*":*^24}')
    print(f'*{"1. 생성하기":^18}*')
    print(f'*{"2. 종료하기":^18}*')
    print(f'{"*":*^24}')


def print_menu2():
    print(f'{"*":*^35}')
    print(f'*{"어떠한 조합으로 생성하시겠습니까?":^18}*')
    print(f'{"*":*^35}')
    print(f'*{"1. 특수문자 + 영문자 + 숫자":^24}*')
    print(f'*{"2. 영문자 + 숫자":^28}*')
    print(f'{"*":*^35}')


def print_menu3():
    print(f'{"*":*^34}')
    print(f'*{"숫자 시드 넘버를 입력해주세요.":^19}*')
    print(f'{"*":*^34}')


def password_generator():
    
    break_flag = False

    while True:
        # 초기화면 문구 보여주기
        print_menu1()

        # 비밀번호 생성기 화면에서 입력 받기
        input_data1 = input("\n숫자 1 또는 2를 입력해주세요. : ")
        print()
    
        if input_data1 == '1':                                  # 1입력: 생성하기 버튼

            while True:

                # 조합 생성 문구 보여주기
                print_menu2()

                # 어떠한 조합으로 생성하시겠습니까? 화면에서 입력 받기
                input_data2 = input("\n숫자 1 또는 2를 입력해주세요. : ")
                print()

                if input_data2 == '1':              # 1입력: 특수문자 + 영문자 + 숫자

                    while True:

                        # 시드 넘버 입력 문구 보여주기
                        print_menu3()

                        # 숫자 시드 넘버를 입력해주세요. 화면에서 입력 받기
                        input_data3 = input('\n0 ~ 100 사이의 정수 시드 넘버를 입력해주세요. : ')
                        print()
                        if input_data3.isdecimal() == True: # 입력 데이터가 숫자일때
                            seed = int(input_data3)

                            if 0 <= seed <= 100:
                                print(f'\n생성된 비밀번호: {generator2(seed)}')
                                print('\n비밀번호 생성을 완료하였습니다. 생성기를 종료합니다.')
                                print()
                                break_flag = True
                                break
                            else:
                                print('\n-----*경고* 0 ~ 100 사이의 정수를 입력해주세요!!-----')
                                print()
                                continue
                
                        else:                               # 입력 데이터가 숫자가 아닐때
                            print('\n-----*경고* 정수를 입력해주세요!!-----')
                            print()
                            continue
                    if break_flag == True:
                        break
            
                elif input_data2 == '2':            # 2입력: 영문자 + 숫자

                    while True:

                        # 시드 넘버 입력 문구 보여주기
                        print_menu3()
                        # 숫자 시드 넘버를 입력해주세요. 화면에서 입력 받기
                        input_data3 = input('\n0 ~ 100 사이의 정수 시드 넘버를 입력해주세요. : ')
                        print()
                        if input_data3.isdecimal() == True:
                            seed = int(input_data3)
                            if 0 <= seed <= 100:
                                print(f'\n생성된 비밀번호: {generator1(seed)}')
                                print('\n비밀번호 생성을 완료하였습니다. 생성기를 종료합니다.')
                                print()
                                break_flag = True
                                break
                            else:
                                print('\n-----*경고* 0 ~ 100 사이의 정수를 입력해주세요!!-----')
                                print()
                                continue
                        else:                               # 입력 데이터가 숫자가 아닐때
                            print('\n-----*경고* 정수를 입력해주세요!!-----')
                            print()
                            continue
                    if break_flag == True:
                        break
                else:
                    print('\n-----*경고* 1 또는 2를 입력하세요!!-----')
                    print()
            if break_flag == True:
                break
        elif input_data1 == '2':                                # 2입력: 종료하기 버튼
            print('\n비밀번호 생성기를 종료합니다.')
            break
        else:                                                   # 다른 글자 입력: 경고 메시지
            print('\n-----*경고* 1 또는 2를 입력하세요!!-----')
            print()
            continue





