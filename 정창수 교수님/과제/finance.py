from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests

while True:

    url = 'https://finance.naver.com/sise/sise_market_sum.naver'
    html = requests.get(url)
    soup = BeautifulSoup(html.text, 'html.parser')
    tbody = soup.find('tbody')

    company_name = []
    for i in range(10):
        company_name.append(tbody.find_all('a')[2*i].string)

    print(f'-------------------------------------')
    print(f'[ 네이버 코스피 상위 10대 기업 목록 ]')
    print(f'-------------------------------------')
    for i in range(10):
          print(f'[{i+1:2}] {company_name[i]}')

    company_num = input('주가를 검색할 기업의 번호를 입력하세요(-1: 종료): ')

    if company_num == '-1':
        break
    else:
        company_num = int(company_num)
        detail_list = []
        for i in soup.find_all('td', {'class' : 'center'})[:10]:
            detail_list.append(i.find('a')['href'][-6:])

        today = []
        for i in range(10):
            today.append('현재가: ' + tbody.find_all('td', {'class' : 'number'})[i*10].string)

        temp_url = 'https://finance.naver.com/item/main.naver?code=' + detail_list[company_num-1]
        temp_html = requests.get(temp_url)
        temp_soup = BeautifulSoup(temp_html.text, 'html.parser')
        dl = temp_soup.find('dl')
        total = [today[0]]
        for i in (4, 5, 6, 8):
            total.append(dl.find_all('dd')[i].string.replace(' ', ': '))
        total

        print(temp_url)
        print(f'종목명: {company_name[company_num-1]}')
        print(f'종목코드: {detail_list[company_num-1]}')
        for i in range(len(total)):
            print(total[i])