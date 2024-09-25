import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('gender.csv', encoding = 'euc_kr')
# df

list_data = []
for i in df['행정구역']:
    list_data.append(i.split(' (')[0].strip(' '))

df['행정구역'] = list_data
# df

def find_index(column_name):
    for i in range(len(df.columns)):
        if df.columns[i] == column_name:
            print(f'idx: {i}')

# print('남자 인덱스 위치')
# find_index('2024년06월_남_0세')
# find_index('2024년06월_남_100세 이상')

# print('여자 인덱스 위치')
# find_index('2024년06월_여_0세')
# find_index('2024년06월_여_100세 이상')

def gender_population(city, gender):
    if gender == 'man':
        gender_df = df[df['행정구역'] == city].iloc[:, 106:207]
    elif gender == 'woman':
        gender_df = df[df['행정구역'] == city].iloc[:, 209:310]

    pop_list = []
    for i in list(gender_df.iloc[0]):
        if type(i) == str:
            pop_list.append(int(i.replace(',', '')))
        else:
            pop_list.append(i)

    return sum(pop_list)

plt.rcParams['font.family'] = 'AppleGothic'
fig, axes = plt.subplots(5, 2, figsize = (10, 10))

# city_list = ['대구광역시', '대구광역시 중구', '대구광역시 동구', '대구광역시 서구', '대구광역시 남구',
#              '대구광역시 북구', '대구광역시 수성구', '대구광역시 달서구', '대구광역시 달성군', '대구광역시 군위군']

city_list = list(df['행정구역'][43:53])

for i in range(5):
    for j in range(2):
        axes[i, j].pie([gender_population(city_list[i*2 + j], 'man'),
                        gender_population(city_list[i*2 + j], 'woman')],
                        labels = ['남성', '여성'],
                        startangle=90, autopct='%.1f%%')
        axes[i, j].set_title(city_list[i*2 + j])

plt.suptitle('대구광역시 구별 남녀 인구 비율', fontsize = 20)
plt.tight_layout()
plt.show()