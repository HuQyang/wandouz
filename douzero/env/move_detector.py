from douzero.env.utils import *
import collections
from .utils import *
from .move_generator_for_detector import MovesGener

# check if move is a continuous sequence
def is_continuous_seq(move):
    i = 0
    while i < len(move) - 1:
        if move[i+1] - move[i] != 1:
            return False
        i += 1
    return True

# return the type of the move
def get_move_type(move, mastercard_list):
    move_size = len(move)
    move_dict = collections.Counter(move)
    mcs = sorted([i for i in move if i in mastercard_list])
    # make sure the regular card is sorted:
    regular_card = sorted([i for i in move if i not in mastercard_list])
    regular_card_counts = collections.Counter(regular_card)
    less_than_two = all(count <= 2 for count in regular_card_counts.values())
    less_than_three = all(count <= 3 for count in regular_card_counts.values())
    if len(regular_card) > 0:
        serial_rank = min(regular_card[0],14-move_size+1)
    else:
        serial_rank = 0

    mdkeys = sorted(move_dict.keys())
    if len(regular_card) > 0:
        single_serial_condition = regular_card[-1]<15 and regular_card[-1]-regular_card[0]<move_size
    else:
        single_serial_condition = False

    mg= MovesGener(move,mcs)

    def has_mastercard(move,mastercard_list):
        return any(card in mastercard_list for card in move)

    def mastercard_count(move,mastercard_list):
        return [i for i in move if i in mastercard_list]

    def is_bomb():
        if len(regular_card) == 0 or len(set(regular_card)) == 1:
            return True
        else:
            return False
    
    def is_7_3_2():
        if mg.gen_type_7_3_2():
            return True
        else:
            return False
    
    def is_8_serial_single():
        if mg.gen_type_8_serial_single(repeat_num=move_size):
            return True
        else:
            return False
    def is_9_serial_pair():
        if mg.gen_type_9_serial_pair(repeat_num=move_size//2):
            return True
        else:
            return False
        
    def is_10_serial_triple():
        if mg.gen_type_10_serial_triple(repeat_num=move_size//3):
            return True
        else:
            return False

    def is_11_serial_3_1():
        if mg.gen_type_11_serial_3_1(repeat_num=move_size//4):
            return True
        else:
            return False
    
    def is_12_serial_3_2():
        if len(mg.gen_type_12_serial_3_2(repeat_num=move_size//5))>0:
            return True
        else:
            return False
    
    def is_13_4_2():
        if mg.gen_type_13_4_2():
            return True
        else:
            return False
    
    def is_14_4_22():
        if mg.gen_type_14_4_22():
            return True
        else:
            return False
    
    
    if move == [20,30] or move == [30,20]:
        return {'type': TYPE_5_KING_BOMB}
    
    if move_size == 0:
        return {'type': TYPE_0_PASS}

    if move_size == 1:
        return {'type': TYPE_1_SINGLE, 'rank': move[0]}

    if move_size == 2:
        if move[0] == move[1] or move[1] in mastercard_list:
            return {'type': TYPE_2_PAIR, 'rank': move[0]}
        else:
            return {'type': TYPE_15_WRONG}

    if move_size == 3:
        if len(move_dict) == 1 :
            return {'type': TYPE_3_TRIPLE, 'rank': move[0]}
        elif has_mastercard(move,mastercard_list):
            if len(set(regular_card)) == 1:
                return {'type': TYPE_3_TRIPLE, 'rank': regular_card[0]}
        else:
            return {'type': TYPE_15_WRONG}

    # if move_size == 4:
    #     if len(move_dict) == 1:
    #         return {'type': TYPE_4_BOMB,  'rank': move[0]}
    #     elif len(move_dict) == 2:
    #         if move[0] == move[1] == move[2] or move[1] == move[2] == move[3]:
    #             return {'type': TYPE_6_3_1, 'rank': move[1]}
    #         else:
    #             return {'type': TYPE_15_WRONG}
    #     else:
    #         return {'type': TYPE_15_WRONG}

    if move_size == 4:
        if has_mastercard(move,mastercard_list):
            if len(set(regular_card)) == 1:
                rank = regular_card[0]
                return {'type': TYPE_4_BOMB,  'rank': rank,'len':4,'bomb':'soft'}
            elif len(set(regular_card)) == 2:
                rank = max(regular_card)
                return {'type': TYPE_6_3_1,  'rank':rank,'len':4}
            elif len(regular_card) == 0:
                rank = max(move)
                return {'type': TYPE_4_BOMB,  'rank': rank,'len':4,'bomb':'pure_mc'}
            else:
                return {'type': TYPE_15_WRONG}
        elif len(move_dict) == 1:
                return {'type': TYPE_4_BOMB,  'rank': move[0],'len':4,'bomb':'hard'}
        elif len(move_dict) == 2:
            if move[0] == move[1] == move[2] or move[1] == move[2] == move[3]:
                return {'type': TYPE_6_3_1, 'rank': move[1]}
            else:
                return {'type': TYPE_15_WRONG}
        else:
            return {'type': TYPE_15_WRONG}

    if is_continuous_seq(move) and move_size >= 5:
        return {'type': TYPE_8_SERIAL_SINGLE, 'rank': move[0], 'len': len(move)}

    if move_size == 5:
        '''
        possible moves:
        1. TYPE_4_BOMB
        2. TYPE_8_SERIAL_SINGLE
        3. TYPE_7_3_2
        '''
        if is_bomb():
            return {'type': TYPE_4_BOMB, 'rank': move[0],'len':move_size}
        elif is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        elif is_7_3_2():
            return {'type': TYPE_7_3_2,'len':move_size}

        else:
            return {'type': TYPE_15_WRONG}

    count_dict = collections.defaultdict(int)
    for c, n in move_dict.items():
        count_dict[n] += 1
    
    if move_size == 6:
        '''
        possible moves:
        1. TYPE_8_SERIAL_SINGLE
        6. TYPE_9_SERIAL_PAIR
        3. TYPE_10_SERIAL_TRIPLE
        4. TYPE_13_4_2
        5. TYPE_4_BOMB
        '''

        if is_bomb():
            return {'type': TYPE_4_BOMB, 'rank': move[0],'len':move_size}
        elif is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        elif is_9_serial_pair():
            return {'type': TYPE_9_SERIAL_PAIR, 'rank': min(regular_card),'len':move_size}
        elif is_10_serial_triple():
            return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': min(regular_card[0],14-3+1),'len':3}
        elif is_13_4_2():
            return {'type': TYPE_13_4_2, 'rank': max(regular_card),'len':move_size}
        else:
            return {'type': TYPE_15_WRONG}
            
        # if has_mastercard(move,mastercard_list):
        #     if len(mcs) == 1:
        #         if len(set(regular_card)) == 5 and single_serial_condition:
        #             return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        #         elif len(set(regular_card)) == 3 and is_continuous_seq(list(set(regular_card))) and less_than_two and MIN_PAIRS==3:
        #             return {'type': TYPE_9_SERIAL_PAIR, 'rank': min(regular_card),'len':move_size}
        #         else:
        #             return {'type': TYPE_15_WRONG}
        #     elif len(mcs) == 2:
        #         if len(set(regular_card)) == 4 and single_serial_condition:
        #             return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        #         elif len(set(regular_card)) == 3 and is_continuous_seq(list(set(regular_card))) and MIN_PAIRS==3:
        #             return {'type': TYPE_9_SERIAL_PAIR, 'rank': regular_card[0], 'len': len(regular_card)}
        #         elif len(set(regular_card)) == 2 :
        #             if is_continuous_seq(list(set(regular_card))) and MIN_TRIPLES==2:
        #                 return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': regular_card[0], 'len': 2}
        #             elif regular_card[0] != regular_card[1]:
        #                 rank = regular_card[1]
        #             elif regular_card[2] != regular_card[3]:
        #                 rank = regular_card[2]                 
        #             else:
        #                 rank = max(regular_card)
        #             return {'type': TYPE_13_4_2, 'rank': rank}
        #         else:
        #             return {'type': TYPE_15_WRONG}
        #     elif len(mcs) == 3:
        #         if len(set(regular_card)) == 3:
        #             if is_continuous_seq(list(set(regular_card))) :
        #                 rank = min(regular_card)
        #                 return {'type': TYPE_9_SERIAL_PAIR, 'rank': rank}
        #             elif single_serial_condition:
        #                 return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        #             else:
        #                 return {'type': TYPE_15_WRONG}
        #         # Note: here only consider 4+2 and serial triple, even though serial pair is also possible
        #         # but 4+2 and serial triple is more likely to be the best move as the initial move
        #         elif len(set(regular_card)) == 2 :
        #             if is_continuous_seq(list(set(regular_card))) and MIN_TRIPLES==2:
        #                 return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': regular_card[0], 'len': 2}
        #             else:
        #                 rank = max(regular_card)
        #                 return {'type': TYPE_13_4_2, 'rank': rank}
        #         elif len(set(regular_card)) == 1:
        #             return {'type': TYPE_4_BOMB, 'rank': regular_card[0],'len':move_size,'bomb':'soft'}
        #         else:
        #             return {'type': TYPE_15_WRONG}
        #     elif len(mcs) == 4:
        #         if len(set(regular_card)) == 1:
        #             return {'type': TYPE_4_BOMB, 'rank': regular_card[0],'len':move_size,'bomb':'soft'}
        #         elif len(set(regular_card)) == 2:
        #             if is_continuous_seq(list(set(regular_card))) and MIN_TRIPLES==2:
        #                 return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': regular_card[0], 'len': 2}
        #             else:
        #                 rank = max(regular_card)
        #                 return {'type': TYPE_13_4_2, 'rank': rank,'len':move_size}
        #         else:
        #             return {'type': TYPE_15_WRONG}
        #     elif len(mcs) == 5:
        #         return {'type': TYPE_4_BOMB, 'rank': regular_card[0],'len':move_size,'bomb':'soft'}
        #     elif len(mcs) == 6:
        #         return {'type': TYPE_4_BOMB, 'rank': move[0],'len':move_size,'bomb':'pure_mc'}
        #     else:
        #         return {'type': TYPE_15_WRONG}

        # elif (len(move_dict) == 2 or len(move_dict) == 3) and count_dict.get(4) == 1 and \
        #         (count_dict.get(2) == 1 or count_dict.get(1) == 2):
        #     return {'type': TYPE_13_4_2, 'rank': move[2]}
        # else:
        #     return {'type': TYPE_15_WRONG}
    
    if move_size == 7:
        '''
        possible moves:
        1. TYPE_8_SERIAL_SINGLE
        2. TYPE_4_BOMB
        '''
        if is_bomb():
            return {'type': TYPE_4_BOMB, 'rank': move[0],'len':move_size}
        elif is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        else:
            return {'type': TYPE_15_WRONG}
        

    if move_size == 8:
        '''
        possible moves:
        1. TYPE_4_BOMB
        2. TYPE_14_4_22
        3. TYPE_8_SERIAL_SINGLE
        4. TYPE_9_SERIAL_PAIR
        5. TYPE_11_SERIAL_3_1
        '''

        if is_bomb():
            return {'type': TYPE_4_BOMB, 'len':move_size}
        elif is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        elif is_9_serial_pair():
            return {'type': TYPE_9_SERIAL_PAIR, 'rank': min(regular_card),'len':move_size//2}
        elif is_11_serial_3_1():
            return {'type': TYPE_11_SERIAL_3_1, 'rank': min(regular_card),'len':move_size//4}
        elif is_14_4_22():
            return {'type': TYPE_14_4_22, 'rank': max(regular_card),'len':move_size}
        else:
            return {'type': TYPE_15_WRONG}

    
    if move_size == 9:
        '''
        possible moves:
        1. TYPE_4_BOMB
        2. TYPE_10_SERIAL_TRIPLE
        3. TYPE_8_SERIAL_SINGLE
        '''
        if is_bomb():
            return {'type': TYPE_4_BOMB, 'rank': regular_card[0],'len':move_size}
        elif is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        elif is_10_serial_triple():
            return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': min(regular_card[0],14-3+1),'len':3}
        else:
            return {'type': TYPE_15_WRONG}
           

    if move_size == 10:
        '''
        possible moves:
        1. TYPE_4_BOMB
        2. TYPE_10_SERIAL_pair
        3. TYPE_8_SERIAL_SINGLE
        4. TYPE_12_SERIAL_3_2
        '''
        if is_bomb():
            return {'type': TYPE_4_BOMB, 'rank': regular_card[0],'len':move_size}
        elif is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        elif is_9_serial_pair():
            return {'type': TYPE_9_SERIAL_PAIR, 'rank': min(regular_card),'len':move_size//2}
        elif is_12_serial_3_2():
            return {'type': TYPE_12_SERIAL_3_2, 'rank': move[0],'len':move_size}
        else:
            return {'type': TYPE_15_WRONG}
    
    if move_size == 11:
        '''
        possible moves:
        1. TYPE_4_BOMB
        2. TYPE_8_SERIAL_SINGLE
        '''

        if is_bomb():
            return {'type': TYPE_4_BOMB, 'rank': regular_card[0],'len':move_size}
        elif is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        else:
            return {'type': TYPE_15_WRONG}
      
    if move_size == 12:
        '''
        possible moves:
        1. TYPE_4_BOMB
        2. TYPE_8_SERIAL_SINGLE
        3. TYPE_9_SERIAL_PAIR
        4. TYPE_10_SERIAL_TRIPLE
        '''

        if is_bomb():
            return {'type': TYPE_4_BOMB, 'rank': move[0],'len':move_size}
        elif is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        elif is_9_serial_pair():
            return {'type': TYPE_9_SERIAL_PAIR, 'rank': min(regular_card),'len':move_size}
        elif is_10_serial_triple():
            return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': min(regular_card[0],14-4+1),'len':4}
        elif is_11_serial_3_1():
            return {'type': TYPE_11_SERIAL_3_1, 'rank': min(regular_card),'len':move_size//4}
        else:
            return {'type': TYPE_15_WRONG}
            
        # if has_mastercard(move,mastercard_list):
            
        #     if len(mcs) + len(regular_card) == move_size and single_serial_condition:
        #         return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        #     # elif len(mcs)==1:
        #     #     if is_continuous_seq(list(set(regular_card))) and len(set(regular_card)) == 4 and less_than_three:
        #     #         return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': min(regular_card[0],11),'len':4}
        #     #     elif is_continuous_seq(list(set(regular_card))) and len(set(regular_card)) == 6 and less_than_two:
        #     #         return {'type': TYPE_9_SERIAL_PAIR, 'rank': min(regular_card[0],move_size-6+1),'len':5}
        #     #     else:
        #     #         return {'type': TYPE_15_WRONG}
            
        #     elif len(mcs)<=8:
        #         if less_than_three and regular_card[-1]-regular_card[0]<move_size//3:
        #             return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': min(regular_card[0],14-4+1),'len':4}
        #         elif less_than_two and regular_card[-1]-regular_card[0]<move_size//2:
        #             return {'type': TYPE_9_SERIAL_PAIR, 'rank': min(regular_card[0],14-6+1),'len':6}
        #         elif len(set(regular_card)) == 1:
        #             return {'type': TYPE_4_BOMB, 'rank': regular_card[0],'len':move_size,'bomb':'soft'}
        #         else:
        #             return {'type': TYPE_15_WRONG}
           
        #     else:
        #         return {'type': TYPE_15_WRONG}
    
    if move_size == 13:
        '''
        possible moves:
        1. TYPE_4_BOMB
        2. TYPE_8_SERIAL_SINGLE

        '''
        if is_8_serial_single():
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': serial_rank,'len':move_size}
        else:
            return {'type': TYPE_15_WRONG}
    
    if move_size == 14:
        '''
        possible moves:
        1. TYPE_4_BOMB
        2. TYPE_9_SERIAL_PAIR
        '''
        if is_9_serial_pair():
            return {'type': TYPE_9_SERIAL_PAIR, 'rank': min(regular_card),'len':7}
        else:
            return {'type': TYPE_15_WRONG}
    
    if move_size == 15:
        '''
        possible moves:
        1. TYPE_10_SERIAL_TRIPLE
        2. TYPE_12_SERIAL_3_2
        '''
        if is_10_serial_triple():
            return {'type': TYPE_10_SERIAL_TRIPLE,'len':5}
        elif is_12_serial_3_2():
            return {'type': TYPE_12_SERIAL_3_2, 'len':3}
        else:
            return {'type': TYPE_15_WRONG}
    
    if move_size == 16:
        '''
        possible moves:
        1. TYPE_9_SERIAL_PAIR
        2. TYPE_11_SERIAL_3_1
        '''
        if is_9_serial_pair():
            return {'type': TYPE_9_SERIAL_PAIR, 'len':8}
        elif is_11_serial_3_1():
            return {'type': TYPE_11_SERIAL_3_1, 'len':4}
        else:
            return {'type': TYPE_15_WRONG}
    
    if move_size == 18:
        '''
        possible moves:
        1. TYPE_10_SERIAL_TRIPLE
        2. TYPE_9_SERIAL_PAIR
        '''
        if is_10_serial_triple():
            return {'type': TYPE_10_SERIAL_TRIPLE, 'len':6}
        elif is_9_serial_pair():
            return {'type': TYPE_9_SERIAL_PAIR, 'len':9}
        else:
            return {'type': TYPE_15_WRONG}
    

    
    if len(move_dict) == count_dict.get(2) and is_continuous_seq(mdkeys):
        return {'type': TYPE_9_SERIAL_PAIR, 'rank': mdkeys[0], 'len': len(mdkeys)}

    if len(move_dict) == count_dict.get(3) and is_continuous_seq(mdkeys):
        return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': mdkeys[0], 'len': len(mdkeys)}

    # Check Type 11 (serial 3+1) and Type 12 (serial 3+2)
    if count_dict.get(3, 0) >= MIN_TRIPLES:
        serial_3 = list()
        single = list()
        pair = list()

        for k, v in move_dict.items():
            if v == 3:
                serial_3.append(k)
            elif v == 1:
                single.append(k)
            elif v == 2:
                pair.append(k)
            else:  # no other possibilities
                return {'type': TYPE_15_WRONG}

        serial_3.sort()
        if is_continuous_seq(serial_3):
            if len(serial_3) == len(single)+len(pair)*2:
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[0], 'len': len(serial_3)}
            if len(serial_3) == len(pair) and len(move_dict) == len(serial_3) * 2:
                return {'type': TYPE_12_SERIAL_3_2, 'rank': serial_3[0], 'len': len(serial_3)}

        if len(serial_3) == 4:
            if is_continuous_seq(serial_3[1:]):
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[1], 'len': len(serial_3) - 1}
            if is_continuous_seq(serial_3[:-1]):
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[0], 'len': len(serial_3) - 1}

    return {'type': TYPE_15_WRONG}


