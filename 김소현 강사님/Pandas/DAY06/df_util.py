# --------------------------------------------------------------
# Series/DataFrame에서 사용되는 사용자 정의 함수들
# --------------------------------------------------------------
# 함수 기능 : DataFrame의 기본 정보와 속성 확인 기능
# 함수 이름 : print_info()
# 매개 변수 : DataFrame 인스턴스 변수명, DataFrame 인스턴스 이름
# 리턴값/반환값 : 없음
# --------------------------------------------------------------
def print_info(df, name):
    print(f'\n[{name}]')
    df.info()
    print(f'[Index] : {df.index}')
    print(f'[Columns] : {df.columns}')
    print(f'[NDim] : {df.ndim}')