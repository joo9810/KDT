from card import Card
from player import Player
from gamedealer import GameDealer

def play_game():
    # 두 명의 player 객체 생성
    player1 = Player("흥부")
    player2 = Player("놀부")

    dealer = GameDealer()

    deck = dealer.make_deck()
    dealer.shuffle_deck() # 덱 섞기

    turn = 1

    while True:
        if turn == 1 or turn >= 3: # 첫 턴 일때

            if turn == 1:
                hand_out = 10 # 첫 턴이면 10장 나눠주기
            else:
                input(f'[{turn}]단계: 다음 단계 진행을 위해 Enter 키를 누르세요!')
                hand_out = 4 # 3턴 이상이면 4장 나눠주기

            player1.add_card_list(deck[:hand_out])
            player2.add_card_list(deck[hand_out:2*hand_out])

            deck = deck[2*hand_out:]

            print(f'============================================================')
            dealer.print_deck(deck, f'카드 나누어 주기: {hand_out}장')

            

            if turn == 1:
                player1.display_two_card_lists()
                player2.display_two_card_lists()
                turn += 1


        if turn == 2:
            input(f'[{turn}]단계: 다음 단계 진행을 위해 Enter 키를 누르세요!')

        print(f'============================================================')
        print(f'[{player1.name}: 숫자가 같은 한쌍의 카드 검사]')
        player1.check_one_pair_card() # 같은 숫자 있는지 검사
        player1.display_two_card_lists()

        print(f'============================================================')
        print(f'[{player2.name}: 숫자가 같은 한쌍의 카드 검사]')
        player2.check_one_pair_card() # 같은 숫자 있는지 검사
        player2.display_two_card_lists()

        turn += 1 # 턴 숫자 추가

        if len(deck) == 0 or len(player1.holding_card_list) == 0 or len(player2.holding_card_list) == 0:
            break
            
if __name__ == '__main__':
    play_game()