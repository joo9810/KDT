'''
    출근 시간대(07:00 ~ 08:59) 노선별 최대 하차 인원 계산
     - 1 ~ 7호선 정보만 출력

'''

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import koreanize_matplotlib


def draw_bar_chart(max_station_info):
    max_station_number = []
    max_station_name = []

    ### 각 지하철 노선별 최대 하차 인원 막대 그래프로 그리기
    for i in range(len(max_station_info)):
        # [0]: 노선번호, [1]: 역이름, [2]: 최대 하차 인원
        print(f'출근 시간대 {max_station_info[i][0]} '
              f'최대 하차역: {max_station_info[i][1]}역, '
              f'하차인원: {max_station_info[i][2]:,}명')

        max_station_name.append(max_station_info[i][0] + ' ' + max_station_info[i][1])
        max_station_number.append(max_station_info[i][2])

    plt.figure(dpi=100)
    plt.bar(range(7), max_station_number)
    plt.xticks(range(7), max_station_name, rotation=80)
    plt.title('출근 시간대 지하철 노선별 최대 하차 인원 및 하차역', size=12)
    plt.tight_layout()
    plt.show()


def get_subway_line_max_number():
    df = pd.read_excel('subway.xls', sheet_name='지하철 시간대별 이용현황',
                       header=[0, 1])

    # 출근 시간대 정보만 추출 후 시간대 정보를 int64로 변경
    # [1]: 호선명, [3]: 역이름, [11]: 7시대 하차, [13]: 8시대 하차
    commute_df = df.iloc[:, [1, 3, 11, 13]]

    for i in [2, 3]:    # [11] -> [2] 7시 하차, [13] -> [3] 8시 하차
        commute_df.iloc[:, i] = commute_df.iloc[:, i].apply(lambda x: x.replace(',', ''))
        commute_df.iloc[:, i] = commute_df.iloc[:, i].astype('int64')
    #print(commute_df.head())

    commute_df.columns = ['호선명', '지하철역', '7시하차', '8시하차']
    #print(commute_df)

    passenger_df = commute_df[['7시하차', '8시하차']].sum(axis=1)
    total_commute_df = pd.concat([commute_df, passenger_df], axis=1)
    total_commute_df.columns= ['호선명', '지하철역', '7시하차', '8시하차', '총하차인원']

    print(tabulate(total_commute_df.head(), headers='keys', tablefmt='psql'))

    # 노선 정보를 리스트로 만듬
    line_list = [str(n) + '호선' for n in range(1, 8)]
    max_station_info =[] # 저장 형식: [[노선번호, 지하철역, 하차인원]]

    for line in line_list:
        line_df = total_commute_df[total_commute_df['호선명'] == line]
        line_max_num = line_df['총하차인원'].max()
        line_max_index = line_df['총하차인원'].idxmax()
        line, line_max_station = total_commute_df.iloc[line_max_index, [0, 1]] # 호선명, 지하철역 리턴
        # [[호선명, 지하철역, 최대하차인원]] 형태로 저장
        max_station_info.append([line, line_max_station, line_max_num])


    # bar 차트 그리기
    draw_bar_chart(max_station_info)


get_subway_line_max_number()
