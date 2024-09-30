class Player:
    def __init__(self, name):
        self.name = name
        self.holding_card_list = list()
        self.open_card_list = list()

    def add_card_list(self, card_list):
        for i in card_list:
            self.holding_card_list.append(i)

    def display_two_card_lists(self):
        print(f'============================================================')
        print(f'[{self.name}] Open card list: {len(self.open_card_list)}')
        for i in self.open_card_list:
            print(i, end = ' ')
        print('\n')
        print(f'[{self.name}] Holding card list: {len(self.holding_card_list)}')
        for i in self.holding_card_list:
            print(i, end = ' ')
        print('\n')

    def check_one_pair_card(self):
        while True:
            changed = False
            for i in range(len(self.holding_card_list)-1):
                for j in range(i+1, len(self.holding_card_list)):
                    if self.holding_card_list[i][3:5] == self.holding_card_list[j][3:5]:
                        self.open_card_list.append(self.holding_card_list.pop(i))
                        self.open_card_list.append(self.holding_card_list.pop(j-1))
                        changed = True
                        break
                        
                if changed == True:
                    break

            if len(self.holding_card_list) == 0:
                break
            elif i==len(self.holding_card_list)-2 and j==len(self.holding_card_list)-1:
                break

if __name__ == '__main__':
    player1 = Player('흥부')
    # player1.add_card_list(['(♠,10)', '(◆,10)'])
    player1.add_card_list(['(♣, A)', '(♥, 4)', '(♠,10)', '(♥, 6)',
                           '(♣, 4)', '(♠, 9)', '(♥,10)', '(◆, K)', '(♠, Q)', '(♠, 7)'])
    player1.check_one_pair_card()
    player1.display_two_card_lists()