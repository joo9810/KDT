import os

# SQLite3 RDBMS
BASE_DIR = os.path.dirname(__file__)
DB_NAME = 'myweb.db'

# DB관련 기능 구현 시 사용할 변수
SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, DB_NAME))
SQLALCHEMY_TRACK_MODIFICATIONS = False