import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

df = pd.read_csv('subwaytime.csv', encoding='utf-8-sig')

def find_subway_population(line):
    df_station = df[df['호선명'] == line][['호선명', '지하철역', 'Unnamed: 11', 'Unnamed: 13']]
    df_station_num = df_station.iloc[:, 2:]
    df_station_num.columns = ['07:00:00~07:59:59', '08:00:00~08:59:59']
    df_station_num = df_station_num.apply(pd.to_numeric)
    max_index = df_station_num.sum(axis = 1).idxmax()
    station_name = df_station.loc[max_index, '지하철역']
    max_value = df_station_num.sum(axis = 1).max()
    return [station_name, max_value]

station_list = ['1호선', '2호선', '3호선', '4호선', '5호선', '6호선', '7호선']

name_list = []
value_list = []

for i in station_list:
    name_list.append(f'{i} {find_subway_population(i)[0]}')
    value_list.append(find_subway_population(i)[1])

print(name_list, value_list)

for i in range(len(station_list)):
    print(f'출근 시간대 {station_list[i]} 최대 하차역: {name_list[i]}역, 하차인원: {value_list[i]:,}명')

plt.figure(figsize = (10, 5))
plt.rcParams['font.family'] = 'AppleGothic' # 윈도우에선 빼주고 실행
plt.bar(name_list, value_list)
plt.title('출근 시간대 지하철 노선별 최대 하차 인원 및 하차역')
plt.tight_layout()
plt.xticks(rotation = 70)
plt.show()