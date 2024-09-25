def div(num1, num2):
    if not num2:
        result = '0으로 나눌 수 없음'
    else:
        result = num1 / num2
    print(f'{num1}/{num2} = {result}')

value1 = div(3, 4)

print(value1)