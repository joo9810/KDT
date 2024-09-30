from card import Card
import random

class GameDealer:
    def __init__(self):
        self.deck = list()
        self.suit_number = 13

    def make_deck(self):
        card_suit = ["♠", "♥", "♣", "◆"]
        card_number = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

        for i in card_suit:
            for j in card_number:
                self.deck.append(Card(i, j).__str__())
        return self.deck
    
    def shuffle_deck(self):
        random.shuffle(self.deck)

    def print_deck(self, deck, title):
        print(f'{title}\n'
              f'------------------------------------------------------------\n'
              f'[GameDealer] 딜러가 가진 카드 수: {len(deck)}')
        for i in range(len(deck)):
            if i != 0 and (i % 13 == 0):
                print()
            print(deck[i], end = ' ')
        print()

if __name__ == '__main__':
    dealer = GameDealer()
    deck = dealer.make_deck()
    dealer.print_deck(deck, '초기 카드 생성')
    random.shuffle(deck)
    dealer.print_deck(deck, '카드 랜덤하게 섞기')

