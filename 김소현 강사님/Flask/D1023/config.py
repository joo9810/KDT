import os

# SQLite3 RDBMS
BASE_DIR = os.path.dirname(__file__)
DB_NAME = 'myweb.db'

# DB관련 기능 구현 시 사용할 변수
DB_SQLITE_URI = f"'sqlite:///{os.path.join(BASE_DIR, DB_NAME)}"

SQLALCHEMY_DATABASE_URI = 'sqlite:////Users/anhyojun/WorkSpace/KDT2/김소현 강사님/Flask/D1023/myweb.db'
SQLALCHEMY_TRACK_MODIFICATIONS = False