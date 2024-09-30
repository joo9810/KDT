import pandas as pd
from tabulate import tabulate

df = pd.read_csv('hollys_branches.csv', encoding = 'utf-8-sig')

while True:
    string = input('검색할 매장의 지역을 입력하세요: ')
    if string == 'quit':
        print('종료 합니다.')
        break
    else:
        split_list = string.split()
        split_list

        boolean_list = []
        for i in df['주소']:
            for j in split_list:
                if (j in i) == True:
                    boolean_list.append(True)
                else:
                    boolean_list.append(False)

        location_list = []
        for i in range(0, len(boolean_list), len(split_list)):
            if sum(boolean_list[i:i+len(split_list)]) == len(split_list):
                location_list.append(int(i/len(split_list)))
        location_list

        select_df = df.iloc[location_list,:]
        select_df.reset_index(inplace=True)
        select_df.index = select_df.index + 1
        final_df = select_df[['매장이름', '주소', '전화번호']]
        if len(final_df) == 0:
            print('검색된 매장이 없습니다.\n')
        else:
            print(f'검색된 매장 수: {len(final_df)}')
            print(tabulate(final_df, headers = 'keys', tablefmt='psql'))