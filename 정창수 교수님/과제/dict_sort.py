country_dict = {'Seoul' : ['South Korea', 'Asia', 9655000],
                'Tokyo' : ['Japan', 'Asia', 14110000],
                'Beijing' : ['China', 'Asia', 21540000],
                'London' : ['United Kingdom', 'Europe', 14800000],
                'Berlin' : ['Germany', 'Europe', 3426000],
                'Mexico City' : ['Mexico', 'America', 21200000]}

while True:
    print('-' * 50)
    print(f'\n1. 전체 데이터 출력\n'
          f'2. 수도 이름 오름차순 출력\n'
          f'3. 모든 도시의 인구수 내림차순 출력\n'
          f'4. 특정 도시의 정보 출력\n'
          f'5. 대륙별 인구수 계산 및 출력\n'
          f'6. 프로그램 종료\n')
    print('-' * 50)
    menu_number = input('메뉴를 입력하세요: ')
    
    if menu_number == '1':
        for i in range(len(country_dict)):
            print(f'{list(country_dict.keys())[i]}: {list(country_dict.values())[i]}')

    elif menu_number == '2':
        sorted_data = sorted(country_dict.items(), key = lambda x : x[0])
        for i in range(len(sorted_data)):
            print(f'{sorted_data[i][0]:11} : {sorted_data[i][1][0]:15} {sorted_data[i][1][1]:8} {sorted_data[i][1][2]:>12,}')

    elif menu_number == '3':
        sorted_data = sorted(country_dict.items(), key = lambda x : x[1][2], reverse=True)
        for i in range(len(sorted_data)):
            print(f'{sorted_data[i][0]:11} : {sorted_data[i][1][2]:>12,}')

    elif menu_number == '4':
        city_name = input('출력할 도시 이름을 입력하세요: ')
        if city_name in country_dict.keys():
            print(f'도시:{city_name}')
            print(f'국가:{country_dict[city_name][0]}, 대륙:{country_dict[city_name][1]}, 인구수:{country_dict[city_name][2]:,}')
        else:
            print(f'도시이름: {city_name}은 key에 없습니다.')
    
    elif menu_number == '5':
        continent_name = input('대륙 이름을 입력하세요(Asia, Europe, America): ')
        total = 0
        for i in range(len(country_dict)):
            if list(country_dict.items())[i][1][1] == continent_name:
                print(f'{list(country_dict.items())[i][0]}: {list(country_dict.items())[i][1][2]:,}')
                total = total + list(country_dict.items())[i][1][2]
        print(f'{continent_name} 전체 인구수: {total:,}')
        
    elif menu_number == '6':
        break