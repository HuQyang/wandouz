# return all moves that can beat rivals, moves and rival_move should be same type
import collections
global global_mastercard_values


def common_handle(moves, rival_move):
    new_moves = list()
    for move in moves:
        if move[0] > rival_move[0]:
            new_moves.append(move)
    return new_moves

def common_handle_serial(moves,rival_move,mastercard_list):
    new_moves = list()

    start_rival_move = 14-len(rival_move)
    for i in range(len(rival_move)):
        if rival_move[i] in mastercard_list:
            continue
        else:
            start_rival_move = rival_move[i]-i
            break

    start_move = 14-len(rival_move)
    for move in moves:
        for i in range(len(move)):
            if move[i] in mastercard_list:
                continue
            else:
                start_move = move[i]-i
                break

        if start_move > start_rival_move:
            new_moves.append(move)
    
    return new_moves

def common_handle_triple_and_pair(moves, rival_move,mastercard_list):
    new_moves = list()
    regular_card_rival = sorted([i for i in rival_move if i not in mastercard_list])
    if len(regular_card_rival) > 0:
        rival_rank = regular_card_rival[0]
    else:
        rival_rank = max(mastercard_list)
    for move in moves:
        regular_card = sorted([i for i in move if i not in mastercard_list])
        if len(regular_card) > 0:
            my_rank = regular_card[0]
        else:
            my_rank = max(mastercard_list)
        if my_rank > rival_rank:
                new_moves.append(move)
    return new_moves

def common_handle_bomb(moves, rival_move,mastercard_list):
    def has_mastercard(move,mastercard_list=mastercard_list):
        return any(card in mastercard_list for card in move)
    new_moves = list()

    regular_card_rival = [i for i in rival_move if i not in mastercard_list]
    for move in moves:
        regular_card = [i for i in move if i not in mastercard_list]
        if len(move) > 4:
            if len(move) > len(rival_move): # compare length
                    new_moves.append(move)
            elif len(move) == len(rival_move):
                if len(regular_card) == 0: # pure mastercard bomb
                    new_moves.append(move)
                elif len(regular_card_rival) == 0:
                    continue
                elif regular_card[0] > regular_card_rival[0]: # soft compare first card (rank)
                    new_moves.append(move)
        elif len(move) == 4 and len(rival_move) == 4:
            if len(regular_card) == 0 and len(regular_card_rival) > 0: # pure mastercard bomb
                new_moves.append(move)
            elif len(regular_card_rival) == 0:
                continue
            elif not has_mastercard(move) and not has_mastercard(rival_move): # both hard bomb
                if regular_card[0] > regular_card_rival[0]: # compare first card (rank)
                    new_moves.append(move)
            elif not has_mastercard(move) and has_mastercard(rival_move): # hard bomb > soft bomb
                new_moves.append(move)
            elif has_mastercard(move) and has_mastercard(rival_move): # both soft bomb 
                if regular_card[0] > regular_card_rival[0]: # compare first card (rank)
                    new_moves.append(move)
        elif move ==[20,30]:
            new_moves.append(move)
    
    return new_moves

def filter_type_1_single(moves, rival_move,mastercard_list):
    return common_handle(moves, rival_move)


def filter_type_2_pair(moves, rival_move,mastercard_list):
    return common_handle(moves, rival_move)


def filter_type_3_triple(moves, rival_move,mastercard_list):
    return common_handle(moves, rival_move)


def filter_type_4_bomb(moves, rival_move,mastercard_list):
    return common_handle_bomb(moves, rival_move,mastercard_list)

# No need to filter for type_5_king_bomb

def filter_type_6_3_1(moves, rival_move,mastercard_list):
    if rival_move[0] in mastercard_list:
        rival_rank = 17
    else:
        rival_rank = rival_move[0]
    new_moves = list()
    for move in moves:
        if move[0] in mastercard_list:
            my_rank = 17
        else:
            my_rank = move[0]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_7_3_2(moves, rival_move,mastercard_list):
    # assert len(rival_move) == 5
    if rival_move[0] in mastercard_list:
        rival_rank = 17
    else:
        rival_rank = rival_move[0]
    new_moves = list()
    for move in moves:
        # my_rank = move[0]
        if move[0] in mastercard_list:
            my_rank = 17
        else:
            my_rank = move[0]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_8_serial_single(moves, rival_move,mastercard_list):
    return common_handle_serial(moves, rival_move,mastercard_list)

def filter_type_9_serial_pair(moves, rival_move,mastercard_list):
    return common_handle_serial(moves, rival_move,mastercard_list)

def filter_type_10_serial_triple(moves, rival_move,mastercard_list):
    return common_handle_serial(moves, rival_move,mastercard_list)

def filter_type_11_serial_3_1(moves, rival_move,mastercard_list):
    new_moves = list()
    serial_count = len(rival_move)//4
    def get_triple_rank(move):
        for i in range(serial_count):
            if move[i*3] in mastercard_list:
                continue
            else:
                return move[i*3]
        return 14 - serial_count + 1
    for move in moves:
        if get_triple_rank(move) > get_triple_rank(rival_move):
            new_moves.append(move)

    return new_moves

def filter_type_12_serial_3_2(moves, rival_move,mastercard_list):
    new_moves = list()
    serial_count = len(rival_move)//5
    def get_triple_rank(move):
        for i in range(serial_count):
            if move[i*3] in mastercard_list:
                continue
            else:
                return move[i*3]
        return 14 - serial_count + 1
    for move in moves:
        if get_triple_rank(move) > get_triple_rank(rival_move):
            new_moves.append(move)

    return new_moves

def filter_type_13_4_2(moves, rival_move,mastercard_list):
    return common_handle(moves, rival_move)

def filter_type_14_4_22(moves, rival_move,mastercard_list):
    return common_handle(moves, rival_move)

