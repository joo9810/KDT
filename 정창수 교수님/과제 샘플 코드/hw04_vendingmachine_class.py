class VendingMachine:
    def __init__(self, input_dict):
        '''
        생성자
        :param input_dict: 초기 자판기 재료량(dict형태)
        '''
        self.input_money = 0
        self.inventory = input_dict


    def choose_menu(self):
        """
        메뉴 출력 및 메뉴 선택 기능
        :return: option
        """
        print("----------------------------------------")
        print(f"  커피 자판기 (잔액:{self.input_money}원)  ")
        print("----------------------------------------")
        print(" 1. 블랙 커피 ")
        print(" 2. 프림 커피 ")
        print(" 3. 설탕 프림 커피")
        print(" 4. 재료 현황")
        print(" 5. 종료")

        option = int(input("메뉴를 선택하세요: "))
        return option

    def update_inventory(self, key, new_value):
        """
        딕셔너리(inventory)의 재료 현황을 업데이트함
        :param key:
        :param new_value:
        :return:
        """
        if key in self.inventory:
            #current_value = self.inventory.get(key)
            self.inventory[key] += new_value
        else:
            print(f'{key}가 물품 목록에 없습니다.')

    # def update_inventory_list(self, keylist, valuelist):
    #
    #     for key, value in zip(keylist, valuelist):
    #         if key in self.inventory:
    #             self.inventory[key] += value

    def print_inventory(self):
        """
        딕셔너리에 저장된 현재 재료 현황을 화면에 출력
        :return:
        """
        print("------------------------------------------------------------------------------------------")
        print("재료 현황", end=': ')
        for key, item in self.inventory.items():
            print(f'{key}: {item} ', end=' ')

        print()
        print("------------------------------------------------------------------------------------------")

    def check_inventory(self, menu):
        """
        커피를 만들기 전에 각 메뉴에 맞는 재료가 남아 있는지 체크
        :param menu:
        :return:
        """
        coffee = self.inventory.get('coffee')
        cream = self.inventory.get('cream')
        sugar = self.inventory.get('sugar')
        water = self.inventory.get('water')
        cup = self.inventory.get('cup')

        if water >= 100 and cup >= 1 and self.input_money >= 300:
            if menu == 1:  # Black coffee
                if coffee >= 30:
                    return True
                else:
                    return False
            elif menu == 2:  # Cream coffee
                if coffee >= 15 and cream >= 15:
                    return True
                else:
                    return False
            elif menu == 3:  # Cream + Sugar coffee
                if coffee >= 10 and cream >= 10 and sugar >= 10:
                    return True
                else:
                    return False
        elif water < 100 or cup < 1:
            return False

    def making_coffee(self, menu):
        """
        커피 제공 및 소모량 업데이트: dictionary update
        """

        if self.check_inventory(menu) == True:
            self.input_money -= 300

            if menu == 1:
                print("블랙 커피를 선택하셨습니다. 잔액: {0}".format(self.input_money))
                self.update_inventory('coffee', -30)
            elif menu == 2:
                print("프림 커피를 선택하셨습니다. 잔액: {0}".format(self.input_money))
                self.update_inventory('coffee', -15)
                self.update_inventory('cream', -15)
            elif menu == 3:
                print("설탕 프림 커피를 선택하셨습니다. 잔액: {0}".format(self.input_money))
                self.update_inventory('coffee', -10)
                self.update_inventory('cream', -10)
                self.update_inventory('sugar', -10)


            # 공통 업데이트 항목
            self.update_inventory('water', -100)
            self.update_inventory( 'cup', -1)
            self.update_inventory('change', 300)

            # 업데이트 항목 모두 출력
            self.print_inventory()
            return True
        else:
            print('재료가 부족합니다.')
            self.print_inventory()
            return False

    def run(self):
        """
        커피 자판기 동작 및 메뉴 호출 함수
        :return:
        """
        self.input_money = int(input("동전을 투입하세요: "))
        while True:
            if self.input_money < 300:
                print(f"투입된 돈 ({self.input_money}원)이 300원보다 작습니다. ")
                print(f'{self.input_money}원을 반환합니다.')
                break

            menu = self.choose_menu()

            if menu < 1 or menu > 5:
                print("잘못된 메뉴입니다. 메뉴를 다시 선택하세요!")
                continue
            elif menu == 4:
                self.print_inventory()
            elif menu == 5:
                print(f'종료를 선택했습니다. {self.input_money}원이 반환됩니다.')
                break
            elif menu >= 1 or menu <= 3:
                if not self.making_coffee(menu):
                    print(f'{self.input_money}원을 반환합니다.')
                    break

        print("-------------------------------")
        print("커피 자판기 동작을 종료합니다.")
        print("-------------------------------")


inventory_dict = {'coffee': 100, 'cream': 100, 'sugar': 100,
                 'water': 500, 'cup': 5, 'change': 0}
coffee_machine = VendingMachine(inventory_dict)
coffee_machine.run()
