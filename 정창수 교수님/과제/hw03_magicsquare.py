while True:
    n = input('홀수차 배열의 크기를 입력하세요: ')

    if n.isnumeric() == True:
        n = int(n)

        if n % 2 == 1:
            magic_df = [[0] * n for _ in range(n)]
            # magic_df = pd.DataFrame(magic_array)
            magic_df

            row = 0
            col = n // 2

            # row, col

            # turn1
            # magic_df.loc[row, col] = 1
            magic_df[row][col] = 1
            magic_df

            # turn2 이후
            for i in range(2, n*n+1):
                count = i
                row -= 1
                col += 1

                if row == -1 and col == n:
                    row += 2
                    col -= 1

                elif row != -1 and col == n:
                    col -= n

                elif row == -1 and col != n:
                    row += n

                # if magic_df.loc[row, col] != 0:
                #     row += 2
                #     col -= 1

                if magic_df[row][col] != 0:
                    row += 2
                    col -= 1
                
                # magic_df.loc[row, col] = count
                magic_df[row][col] = count
            
            print(f'Magic Square ({n}x{n})')
            for j in range(n):
                for k in range(n):
                    print(f'{magic_df[j][k]:2}', end = ' ')
                    if k == n-1:
                        print()
            break
                    
        else:
            print('짝수를 입력하였습니다. 다시 입력하세요.')

    else:
        print('정수를 입력하세요')

