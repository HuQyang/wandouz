from douzero.env.utils import MIN_SINGLE_CARDS, MIN_PAIRS, MIN_TRIPLES, select
import collections
import itertools
from itertools import combinations_with_replacement, combinations, permutations
from functools import lru_cache
from collections import Counter
from typing import Tuple, List

class MovesGener(object):
    """
    This is for generating the possible combinations
    """
    ALL_RANKS = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,30]

    def __init__(self, cards_list,mastercard_list):
        self.cards_list = cards_list
        self.cards_dict = collections.defaultdict(int)
        

        self.mastercard_list = mastercard_list
        self.avail_mastercard = [card for card in cards_list if card in mastercard_list and card < 20]

        for i in self.cards_list:
            self.cards_dict[i] += 1
            
        self.mastercard_count = sum(self.cards_dict[m] for m in mastercard_list)

        self.single_card_moves = []
        self.gen_type_1_single()
        self.pair_moves = []
        self.gen_type_2_pair()
        self.triple_cards_moves = []
        self.gen_type_3_triple()
        self.bomb_moves = []
        self.gen_type_4_bomb()
        self.final_bomb_moves = []
        self.gen_type_5_king_bomb()
        self.ALL_RANKS = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,30]

    def _gen_serial_moves(self, cards, min_serial, repeat=1, repeat_num=0):
        if 0 < repeat_num < min_serial:
            repeat_num = min_serial
        
        card_count = {}
        for card in cards:
            card_count[card] = card_count.get(card, 0) + 1

        regular_cards = sorted([card for card in set(cards) if card not in self.mastercard_list])
        
        # Get count of each regular card
        regular_card_counts = {card: self.cards_dict[card] for card in regular_cards}
        
        # Check if any regular card has more than 2 instances
        has_more_than_two = any(count > 2 for count in regular_card_counts.values())
        
        if not regular_cards:
            return []  

        min_value = 3
        max_value = 14
        moves = []
        seen_moves = set()  # Track unique moves
        
        for start_value in range(max(min_value, min(regular_cards) - self.mastercard_count), max_value + 1):
            for length in range(min_serial, len(regular_cards) + self.mastercard_count + 1):
                if start_value + length - 1 > max_value:
                    continue
                    
                if repeat_num > 0 and length != repeat_num:
                    continue
                
                target_values = list(range(start_value, start_value + length))

                # Check if we can form this sequence
                card_used = {card: 0 for card in card_count}
                sequence_possible = True
                actual_sequence = []

                for target in target_values:
                    # Check how many of this regular card we have
                    natural_count = self.cards_dict[target]
                    can_use = min(natural_count, repeat)

                    # Use natural cards first
                    if can_use > 0:
                        actual_sequence.extend([target] * can_use)
                        card_used[target] = can_use
                        
                    remaining = repeat - can_use
                    available_mc_count = sum(self.cards_dict[mc] - card_used.get(mc, 0) for mc in self.mastercard_list)
                    # If we don't have enough mastercards left, this sequence is impossible
                    if remaining > available_mc_count:
                        sequence_possible = False
                        break
                        
                    if remaining > 0:
                        # Try to use mastercards
                        masters_found = 0
                        for mc in self.mastercard_list:
                            # Check how many of this mastercard we can still use
                            available = self.cards_dict.get(mc, 0) - card_used.get(mc, 0)
                            use_count = min(available, remaining - masters_found)
                            if use_count > 0:
                                actual_sequence.extend([mc] * use_count)
                                card_used[mc] = card_used.get(mc, 0) + use_count
                                masters_found += use_count
                                
                            if masters_found >= remaining:
                                break

                # # If we were able to form the sequence
                if sequence_possible and len(actual_sequence) == length * repeat:
                    if self.count_mastercards(actual_sequence) > self.mastercard_count:
                        continue

                    used_mc_counts = collections.Counter(mc for mc in actual_sequence if mc in self.mastercard_list)
                    avail_mc_counts = collections.Counter(self.avail_mastercard)

                    # make sure not use more mastercards than available
                    if self.check_allcard_usage(actual_sequence):
                    
                        move_tuple = tuple(actual_sequence)
                        if move_tuple not in seen_moves:
                            seen_moves.add(move_tuple)
                            
                            moves.append(actual_sequence)
                            unused_mc = []
                            for card, count in avail_mc_counts.items():
                                if card not in used_mc_counts or used_mc_counts[card] < count:
                                    unused_instances = count - used_mc_counts.get(card, 0)
                                    unused_mc.extend([card] * unused_instances)
                            
                            # For each used mastercard
                            for used_mc in used_mc_counts:
                                # Try replacing it with each unused mastercard
                                for new_mc in set(unused_mc):  # Use set to avoid duplicate replacements
                                    new_sequence = [new_mc if x == used_mc else x for x in actual_sequence]
                                    if self.count_mastercards(new_sequence) > self.mastercard_count:
                                        continue
                                    move_tuple = tuple(new_sequence)
                                    if move_tuple not in seen_moves and self.check_allcard_usage(new_sequence):
                                        seen_moves.add(move_tuple)
                                        moves.append(new_sequence)
        
        return moves


    def gen_type_1_single(self):
        self.single_card_moves = []
        for i in set(self.cards_list):
            self.single_card_moves.append([i])
        return self.single_card_moves

    def gen_type_2_pair(self):
        self.pair_moves = []
        for k, v in self.cards_dict.items():
            if k > 17:
                continue
            if k in self.mastercard_list:
                continue
            if v >= 2:
                self.pair_moves.append([k, k])
            
            if v == 1:
                for m in self.mastercard_list:
                    if self.cards_dict[m] > 0 and k < 20:
                        move = [k, m]
                        self.pair_moves.append(move)

        # Pure mastercard pairs
        if self.mastercard_count >= 2:
            seen_moves = []
            for i in range(len(self.avail_mastercard)):
                for j in range(len(self.avail_mastercard)):
                    if i != j:
                        move = [self.avail_mastercard[i], self.avail_mastercard[j]]
                        if move not in seen_moves:
                            seen_moves.append(move)
                            self.pair_moves.append(move)

        return self.pair_moves

    def gen_type_3_triple(self):
        self.triple_cards_moves = []
        for k, v in self.cards_dict.items():
            if k > 17:
                continue
            if k in self.mastercard_list:
                continue  # Skip mastercards, handle them separately
            if v >= 3:
                self.triple_cards_moves.append([k, k, k])
            if v >= 2:
                for m in set(self.avail_mastercard):
                    if self.cards_dict[m] > 0:
                        move = [k, k, m]
                        self.triple_cards_moves.append(move)
            
            moves_with_mc = self.unique_unordered_pairs(self.avail_mastercard, 2)

            for item in moves_with_mc:
                move = [k] + list(item)
                if self.check_allcard_usage(move):
                    self.triple_cards_moves.append(move)
            
            # pure mastercard triples
        if self.mastercard_count >= 3:
            seen_m = []
            for base_m in self.avail_mastercard:
                if base_m not in seen_m:
                    seen_m.append(base_m)
                    base_m_index = self.avail_mastercard.index(base_m)
                    rest_card = self.avail_mastercard[:base_m_index] + self.avail_mastercard[base_m_index+1:]
                    candidates = self.unique_unordered_pairs(rest_card, 2)
                    for item in candidates:
                        move = [base_m]+list(item)
                        if self.check_allcard_usage(move):
                            self.triple_cards_moves.append(move)

        return self.triple_cards_moves

    # def gen_type_4_bomb(self):
    #     self.bomb_moves = []
    #     for k, v in self.cards_dict.items():
    #         if v == 4:
    #             self.bomb_moves.append([k, k, k, k])
    #     return self.bomb_moves

    def gen_type_4_bomb(self):

        self.bomb_moves = []
        seen_moves = set()  # Track unique moves
        master = self.mastercard_list
        
        # For each regular card value
        for k, v in self.cards_dict.items():
            if k in master or k > 17:
                continue  

            if v == 4:
                seen_moves.add(tuple([k,k,k,k]))
                self.bomb_moves.append([k,k,k,k])
                
            # For each regular card value, try combinations with mastercards
            if v >= 1 :  # Need at least one natural card
                base_move = [k] * v
                candidate_master = self.avail_mastercard
                
                # Try each possible combination of mastercards
                for i in range(1, len(candidate_master) + 1):
                    for combo in combinations(candidate_master, i):
                        move = base_move + list(combo)
                        if len(move) >= 4:
                            # Convert move to tuple for hashing
                            move_tuple = tuple(sorted(move))
                            if move_tuple not in seen_moves:
                                seen_moves.add(move_tuple)
                                self.bomb_moves.append(move)
            
                    
        # Handle pure mastercard combinations
        
        if len(self.avail_mastercard) >= 4:
            # Generate all possible combinations of 4 or more mastercards
            unique_mastercards = list(set(self.avail_mastercard))
            for mc in unique_mastercards:

                current_mastercard = self.avail_mastercard.copy()
                current_mastercard.remove(mc)
                for length in range(3, len(self.avail_mastercard)):
                    for combo in combinations(current_mastercard, length):
                        move = [mc]+list(combo)

                        move_tuple = tuple((move))
                        if move_tuple not in seen_moves and self.check_allcard_usage(move):
                            seen_moves.add(move_tuple)
                            self.bomb_moves.append(move)
            
        return self.bomb_moves

    
    def gen_type_5_king_bomb(self):
        self.final_bomb_moves = []
        if 20 in self.cards_list and 30 in self.cards_list:
            self.final_bomb_moves.append([20, 30])
        return self.final_bomb_moves

    def gen_type_6_3_1(self):
        result = []

        for t in self.single_card_moves:
            for i in self.triple_cards_moves:
                if t[0] != i[0]:
                    move = i+t
                    if self.check_allcard_usage(move):
                        result.append(move)
        
        return result

    def gen_type_7_3_2(self):
        result = list()
        for t in self.pair_moves:
            for i in self.triple_cards_moves:
                if t[0] != i[0]:
                    move = i+t # 3+2 not 2+3
                    if self.check_allcard_usage(move):
                        result.append(move)

        return result

    def gen_type_8_serial_single(self, repeat_num=0):
        return self._gen_serial_moves(self.cards_list, MIN_SINGLE_CARDS, repeat=1, repeat_num=repeat_num)

    def gen_type_9_serial_pair(self, repeat_num=0):
        single_pairs = list()

        for item in self.pair_moves:
            if item[0] not in self.mastercard_list:
                single_pairs.append(item[0])

        return self._gen_serial_moves(single_pairs, MIN_PAIRS, repeat=2, repeat_num=repeat_num)

    def gen_type_10_serial_triple(self, repeat_num=0):
        single_triples = list()

        for item in self.triple_cards_moves:
            if item[0] not in self.mastercard_list:
                single_triples.append(item[0])

        return self._gen_serial_moves(single_triples, MIN_TRIPLES, repeat=3, repeat_num=repeat_num)

    def gen_type_11_serial_3_1(self, repeat_num=0):
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_1_moves = list()
        seen_moves = list()

        for s3 in serial_3_moves:  # s3 is like [3,3,3,4,4,4]
            s3_set = set(s3)
            s3_len = len(s3) // 3 
            new_cards = [i for i in self.cards_list if i not in s3_set]

            # Get any s3_len items from cards
            subcards = select(new_cards, s3_len)

            for i in subcards:
                if s3 + i not in seen_moves:
                    seen_moves.append(s3 + i)
                    if self.check_allcard_usage(s3 + i):
                        serial_3_1_moves.append(s3 + i)

        return list(k for k, _ in itertools.groupby(serial_3_1_moves))

    def gen_type_12_serial_3_2(self, repeat_num=0):
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_2_moves = list()
        # pair_set = sorted([pair[0] for pair in self.pair_moves if pair[0] not in self.mastercard_list])

        for s3 in serial_3_moves:
            s3_set = set(s3)
            s3_len = len(s3) // 3 

            pair_candidates = [i for i in self.pair_moves if i[0] not in s3_set]
            # Get any s3_len items from cards
            subcards = select(pair_candidates, s3_len)

            for i in subcards:
                pair_tail = []
                for k in i:
                    pair_tail = pair_tail+k
                
                if len(pair_tail)==s3_len*2 and self.check_allcard_usage(s3 +pair_tail):
                    serial_3_2_moves.append((s3 +pair_tail))
        
        return serial_3_2_moves

    def gen_type_13_4_2(self):
        result = list()
        for fc in self.bomb_moves:
            if len(fc) == 4:
                # Only include cards that have count > 0 and are not the bomb card (fc[0])
                cards_list = [k for k in self.cards_dict.keys() if k != fc[0] and self.cards_dict[k] > 0]

                subcards = select(cards_list, 2)
                for i in subcards:
                    if self.check_allcard_usage(fc + i):
                        result.append(fc + i)
        return list(k for k, _ in itertools.groupby(result))

    def gen_type_14_4_22(self):
        result = list()

        for fc in self.bomb_moves:
            if len(fc) == 4:
                pair_moves = [i for i in self.pair_moves if i[0] != fc[0]]

                subcards = select(pair_moves, 2)
                for i in subcards:
                    if i[0][0] != i[1][0]:
                        if self.check_allcard_usage(fc + list(i[0]) + list(i[1])):
                            result.append(fc + list(i[0]) + list(i[1]))
        return result

    # generate all possible moves from given cards
    def gen_moves(self):
        moves = []
        moves.extend(self.gen_type_1_single())
        moves.extend(self.gen_type_2_pair())
        moves.extend(self.gen_type_3_triple())
        moves.extend(self.gen_type_4_bomb())
        moves.extend(self.gen_type_5_king_bomb())
        moves.extend(self.gen_type_6_3_1())
        moves.extend(self.gen_type_7_3_2())
        moves.extend(self.gen_type_8_serial_single())
        moves.extend(self.gen_type_9_serial_pair())
        moves.extend(self.gen_type_10_serial_triple())
        moves.extend(self.gen_type_11_serial_3_1())
        moves.extend(self.gen_type_12_serial_3_2())
        moves.extend(self.gen_type_13_4_2())
        moves.extend(self.gen_type_14_4_22())
        return moves
    
    @lru_cache(maxsize=None)
    def generate_moves_from_counts(self, hand_counts: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
        """
        输入 hand_counts：长度 = len(ALL_RANKS) 的 counts tuple，
        输出 next_states：每个合法动作减去后得到的 counts tuple。
        """
        # 1) 把 counts tuple “解码”成列表，仅做一次
        hand_list = []
        for r, cnt in zip(self.ALL_RANKS, hand_counts):
            hand_list += [r] * cnt

        # 2) 调用原逻辑生成 Counter 列表
        moves = self.gen_moves()

        # 3) 把每个 move 转成 counts tuple，并去掉非法结果
        next_states = []
        for mv in moves:
            mv_cnt = mv if isinstance(mv, Counter) else Counter(mv)
            new_counts = tuple(
                hc - mv_cnt.get(r, 0)
                for hc, r in zip(hand_counts, self.ALL_RANKS)
            )
            if all(x >= 0 for x in new_counts):
                next_states.append(new_counts)

        # 4) 去重并返回 tuple
        unique = set(next_states)
        return tuple(unique)

    def unique_unordered_pairs(self,lst,num):
        return list({tuple(sorted(pair)) for pair in itertools.combinations(lst, num)})
    
    def count_mastercards(self,move):
        count = 0
        for card in move:
            if card in self.mastercard_list:
                count += 1
        return count

    def check_mastercard_usage(self,move):
        used_mc_counts = collections.Counter(mc for mc in move if mc in self.mastercard_list)
        avail_mc_counts = collections.Counter(self.avail_mastercard)
        for key in used_mc_counts.keys():
            if used_mc_counts[key] > avail_mc_counts[key]:
                return False
        return True

    def check_allcard_usage(self,move):
        move_dict = collections.Counter(move)
        handcard_dict = collections.Counter(self.cards_list)
        for key in move_dict.keys():
            if move_dict[key] > handcard_dict[key]:
                return False
        return True