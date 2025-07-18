from douzero.env.move_generator_for_detector import MovesGener
import collections

def print_moves(moves, move_type):
    """Helper function to print moves in a readable format"""
    print(f"\n=== {move_type} ===")
    if not moves:
        print("No valid moves")
        return
    for move in (moves):
        print(move)

def test_move_generator():
    # Test case 1: Basic hand with mastercards 6 and 7
    print("\nTest Case 1: Hand with mastercards 6 and 7")
    # Note: When a card is a mastercard, it should appear in both cards_list and mastercards list
    # cards = [3,3,3,4,4,5,5,5,5,6,6,7,7,7,8,8,9,12,14,14,17,17]  # Regular cards
    # mastercards = [3, 7]  # 6 and 7 are mastercards
    cards = [6, 6, 11, 11, 11, 12, 12, 12, 13, 13]
    mastercards = [12,14]
    cards = [3, 3, 3, 4, 5, 5, 7, 8, 9, 10, 12, 13, 13, 17, 17, 17, 20]
    mastercards = [5,8]
    cards = [3, 4, 5, 6, 7, 7, 8, 8, 8, 9, 9, 10, 11, 11, 12, 12, 13, 13, 14, 17]
    mastercards = [5, 8]

    cards = [4, 4, 5, 6, 7, 7, 8, 10, 11, 14, 14, 17,30]
    mastercards = [11, 6]

    cards = [3, 3, 5, 5, 7, 8, 8, 9, 9, 10, 10, 10, 13, 14, 17, 17, 17]
    mastercards = [14, 12]

    cards = [3, 3, 4, 4, 5, 6, 6, 6, 9, 10, 10, 11, 11, 11, 12, 12, 12, 14, 17, 17]
    mastercards = [6, 12]

    cards = [3, 3, 4, 5, 7, 7, 8, 9, 10, 10, 10, 11, 12, 13, 14, 17, 17]
    mastercards = [13, 11]

    cards = [3, 3, 4, 5, 5, 6, 10, 10, 13, 14, 17, 17]
    mastercards = [9, 13]

    cards =[3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 9, 9, 11, 11, 11, 11, 12, 12, 13, 13]
    mastercards = [11, 12]

    cards = [3, 3, 3,  5, 5, 6,7, 7,7,7, 9,10,14,14,14,17, 20,6,8,10]
    mastercards = [4, 8]

    cards  =[3, 7, 12, 4, 4, 4, 5, 5, 12, 6, 8, 11]
    mastercards = [7, 12]
    print(f"Hand: {sorted(cards)}")
    print(f"Mastercards: {mastercards}")
    
    moves_gener = MovesGener(cards, mastercards)
    # print_moves(moves_gener.gen_moves(), "all moves")
    # Test single cards
    # print_moves(moves_gener.gen_type_1_single(), "Singles")
    
    # Test pairs
    # print_moves(moves_gener.gen_type_2_pair(), "Pairs")
    
    # Test triples
    # print_moves(moves_gener.gen_type_3_triple(), "Triples")
    
    # # Test bombs
    # print_moves(moves_gener.gen_type_4_bomb(), "Bombs")
    
    # # Test king bomb
    # print_moves(moves_gener.gen_type_5_king_bomb(), "King Bomb")
    
    # # Test 3+1
    # print_moves(moves_gener.gen_type_6_3_1(), "3+1")
    
    # # # Test 3+2
    # print_moves(moves_gener.gen_type_7_3_2(), "3+2")
    
    # Test serial singles
    # print_moves(moves_gener.gen_type_8_serial_single(), "Serial Singles")
    
    # # Test serial pairs
    # print_moves(moves_gener.gen_type_9_serial_pair(repeat_num=0), "Serial Pairs")
    
    # # Test serial triples
    print_moves(moves_gener.gen_type_10_serial_triple(repeat_num=3), "Serial Triples")
    
    # # Test serial 3+1
    print_moves(moves_gener.gen_type_11_serial_3_1(repeat_num=3), "Serial 3+1")
    
    # # # Test serial 3+2
    # print_moves(moves_gener.gen_type_12_serial_3_2(repeat_num=0), "Serial 3+2")
    
    # # Test 4+2
    # print_moves(moves_gener.gen_type_13_4_2(), "4+2")
    
    # # Test 4+2+2
    # print_moves(moves_gener.gen_type_14_4_22(), "4+2+2")

    # Test case 2: Special hand focusing on bomb combinations
    # print("\nTest Case 2: Special hand for testing bomb combinations")
    # cards = [5,5,5,6,6,6,7,7,7]  # Three 5s, three 6s (mastercard), three 7s (mastercard)
    # mastercards = [6, 7]
    
    # print(f"Hand: {sorted(cards)}")
    # print(f"Mastercards: {mastercards}")
    
    # moves_gener = MovesGener(cards, mastercards)
    # print_moves(moves_gener.gen_type_4_bomb(), "Bombs with multiple mastercards")

    # # Test case 3: Testing pure mastercard combinations
    # print("\nTest Case 3: Testing pure mastercard combinations")
    # cards = [6,6,6,7,7,7]  # All cards are mastercards
    # mastercards = [6, 7]
    
    # # print(f"Hand: {sorted(cards)}")
    # # print(f"Mastercards: {mastercards}")
    
    # moves_gener = MovesGener(cards, mastercards)
    # print_moves(moves_gener.gen_type_4_bomb(), "Pure mastercard bombs")
    # print_moves(moves_gener.gen_type_3_triple(), "Pure mastercard triples")
    # print_moves(moves_gener.gen_type_2_pair(), "Pure mastercard pairs")
    # print_moves(moves_gener.gen_type_8_serial_single(), "Serial singles with mastercards")
    # print_moves(moves_gener.gen_type_9_serial_pair(), "Serial pairs with mastercards")

    # # # Test case 4: Testing serial moves with mastercards
    # print("\nTest Case 4: Testing serial moves with mastercards")
    # cards = [6,7,7,8,8, 9, 12, 14]  # Sequential cards with 6,7 being mastercards
    # mastercards = [6, 7]
    
    # print(f"Hand: {sorted(cards)}")
    # print(f"Mastercards: {mastercards}")
    
    # moves_gener = MovesGener(cards, mastercards)
    # print_moves(moves_gener.gen_type_8_serial_single(), "Serial singles with mastercards")
    # print_moves(moves_gener.gen_type_9_serial_pair(), "Serial pairs with mastercards")

if __name__ == "__main__":
    test_move_generator() 