import pymysql
import pandas as pd

def make_sql_df(query, db):
    conn = pymysql.connect(host = 'localhost', user = 'joo9810', password = 'signcity114',
                        db = db, charset = 'utf8')
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(query)
    rows = cur.fetchall() # 모든 데이터를 가져옴

    df = pd.DataFrame(rows) # DataFrame 형태로 변환

    cur.close()
    conn.close() # 데이터베이스 연결 종료

    return df

def make_table(query, db):
    conn = pymysql.connect(host = 'localhost', user = 'joo9810', password = 'signcity114',
                        db = db, charset = 'utf8')
    cur = conn.cursor()

    cur.execute(query)
    conn.commit()
    print('Table 생성 완료')

    cur.close()
    conn.close() # 데이터베이스 연결 종료

    