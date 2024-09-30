class VendingMachine:
    def __init__(self, input_dict): # self : 객체 자신을 구분하는 변수
        self.input_money = 0
        self.inventory = input_dict

    def run(self):
        money = input('동전을 투입하세요: ')

        if money.isdecimal() == True:
            self.input_money += int(money)
            count = 1 # 입력한 개수 표시
        else:
            return print('정수를 입력하세요')

        while True:
        
            if count == 1 and self.input_money < 300: # 첫 투입때 300원 보다 작은 경우
                print(f'투입된 돈 ({self.input_money}원)이 300원보다 작습니다.\n'
                    f'{self.input_money}원을 반환합니다.\n'
                    f'-------------------------------\n'
                    f'커피 자판기 동작을 종료합니다.\n'
                    f'-------------------------------')
                break

            elif count >= 2 and self.input_money < 300: # 커피를 뽑아먹고 나서 300원 보다 작은 경우
                print(f'잔액 ({self.input_money}원)이 300원보다 작습니다.\n'
                    f'{self.input_money}원을 반환합니다.\n'
                    f'-------------------------------\n'
                    f'커피 자판기 동작을 종료합니다.\n'
                    f'-------------------------------')
                break

            else:
                print(f'----------------------------------------\n'
                    f'  커피 자판기 (잔액:{self.input_money}원)\n'
                    f'----------------------------------------\n'
                    f'1. 블랙 커피\n'
                    f'2. 프림 커피\n'
                    f'3. 설탕 프림 커피\n'
                    f'4. 재료 현황\n'
                    f'5. 종료')
                
                input_menu = input('메뉴를 선택하세요: ')

                if input_menu in ('1', '2', '3', '4', '5'):

                    if input_menu == '1' and self.inventory_check(30, 0, 0): # 블랙 커피
                        self.input_money -= 300

                        self.inventory_update(30, 0, 0) # 커피 30, 프림 0, 설탕 0

                        print(f'블랙 커피를 선택하셨습니다. 잔액: {self.input_money}')
                        self.display_current_inventory()
                        count += 1

                    elif input_menu == '2' and self.inventory_check(15, 15, 0): # 프림 커피
                        self.input_money -= 300

                        self.inventory_update(15, 15, 0) # 커피 15, 프림 15, 설탕 0

                        print(f'프림 커피를 선택하셨습니다. 잔액: {self.input_money}')
                        self.display_current_inventory()
                        count += 1

                    elif input_menu == '3' and self.inventory_check(10, 10, 10): # 설탕 프림 커피
                        self.input_money -= 300

                        self.inventory_update(10, 10, 10) # 커피 10, 프림 10, 설탕 10

                        print(f'설탕 프림 커피를 선택하셨습니다. 잔액: {self.input_money}')
                        self.display_current_inventory()
                        count += 1

                    elif input_menu == '4': # 재료 현황
                        self.display_current_inventory()

                    elif input_menu == '5': # 종료
                        print(f'종료를 선택했습니다. {self.input_money}원이 반환됩니다.\n'
                            f'-------------------------------\n'
                            f'커피 자판기 동작을 종료합니다.\n'
                            f'-------------------------------\n\n'
                            f'Process finished with exit code 0')
                        break

                    else:
                        print('재료가 부족합니다.')
                        self.display_current_inventory()
                        print(f'{self.input_money}원을 반환합니다.\n'
                            f'-------------------------------\n'
                            f'커피 자판기 동작을 종료합니다.\n'
                            f'-------------------------------')
                        break
        
                else:
                    print('올바른 메뉴 번호를 입력하세요')


    def display_current_inventory(self): # 남은 재고 출력해주는 메서드
        print('-' * 90)
        print(f"재료 현황: coffee: {self.inventory['coffee']}  cream: {self.inventory['cream']}  "
                f"sugar: {self.inventory['sugar']}  water: {self.inventory['water']}  "
                f"cup: {self.inventory['cup']}  change: {self.inventory['change']}")
        print('-' * 90)

    def inventory_update(self, coffee, cream, sugar): # 재고 업데이트해주는 메서드
        self.inventory.update({'coffee' : self.inventory['coffee'] - coffee,
                                'cream' : self.inventory['cream'] - cream,
                                'sugar' : self.inventory['sugar'] - sugar,
                                'water' : self.inventory['water'] - 100,
                                'cup' : self.inventory['cup'] - 1,
                                'change' : self.inventory['change'] + 300})
        
    def inventory_check(self, coffee_req, cream_req, sugar_req): # 재고가 남아있는지 확인하는 메서드
        coffee = self.inventory['coffee']
        cream = self.inventory['cream']
        sugar = self.inventory['sugar']
        water = self.inventory['water']
        cup = self.inventory['cup']

        if coffee >= coffee_req and cream >= cream_req and sugar >= sugar_req and water >= 100 and cup >= 1:
            return True
        else:
            return False
        

# VendingMachine 객체 생성
inventory_dict = {'coffee': 100, 'cream': 100, 'sugar': 100,
                  'water': 500, 'cup': 5, 'change': 0}
coffee_machine = VendingMachine(inventory_dict)
coffee_machine.run()    # VendingMachine 동작 메서드