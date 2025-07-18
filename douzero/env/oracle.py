import collections
from typing import List, Dict, Counter, Iterator, Tuple
from itertools import combinations
import time
# --- 常量定义 ---
CARD_VALUE_2 = 17
BLACK_JOKER_VALUE = 20
RED_JOKER_VALUE = 30
ALL_RANKS = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,30]

CardCounter = Counter[int]
Memo = Dict[Tuple[int, ...], int]

# ===================================================================
# 动作生成器 (已补全)
# ===================================================================

def _generate_moves(cards: CardCounter) -> Iterator[CardCounter]:
    """
    一个生成器，用于枚举给定手牌所有可能的第一步出法。
    此版本已补全所有牌型。
    """
    
    # --- 基础牌型 (单张、对子、三张、炸弹) ---
    for r, c in cards.items():
        if c >= 1: yield collections.Counter({r: 1})
        if c >= 2: yield collections.Counter({r: 2})
        if c >= 3: yield collections.Counter({r: 3})
        if c >= 4: yield collections.Counter({r: 4})
    if cards.get(BLACK_JOKER_VALUE, 0) and cards.get(RED_JOKER_VALUE, 0):
        yield collections.Counter({BLACK_JOKER_VALUE: 1, RED_JOKER_VALUE: 1})

    # --- 带牌组合 (三带X, 四带X) ---
    for r_main, c_main in cards.items():
        # 三带X
        if c_main >= 3:
            remaining_after_trio = cards.copy(); remaining_after_trio.subtract({r_main: 3})
            for r_kicker in remaining_after_trio:
                yield collections.Counter({r_main: 3, r_kicker: 1}) # 三带一
            for r_kicker, c_kicker in remaining_after_trio.items():
                if c_kicker >= 2:
                    yield collections.Counter({r_main: 3, r_kicker: 2}) # 三带二
        # 四带X
        if c_main >= 4:
            remaining_after_quad = cards.copy(); remaining_after_quad.subtract({r_main: 4})
            single_kickers = [r for r, c in remaining_after_quad.items() for _ in range(c)]
            if len(single_kickers) >= 2:
                for combo in combinations(single_kickers, 2): # 四带两单
                    yield collections.Counter({r_main: 4}) + collections.Counter(combo)
            pair_kickers = [r for r, c in remaining_after_quad.items() if c >= 2]
            if len(pair_kickers) >= 2:
                for combo in combinations(pair_kickers, 2): # 四带两对
                    yield collections.Counter({r_main: 4, combo[0]: 2, combo[1]: 2})
                    
    # --- 连续牌型 (顺子、连对、飞机) ---
    ranks_no_2 = sorted([r for r in cards if r < CARD_VALUE_2])
    # 顺子 (>=5张)
    for length in range(5, len(ranks_no_2) + 1):
        for i in range(len(ranks_no_2) - length + 1):
            start_rank = ranks_no_2[i]
            if ranks_no_2[i+length-1] == start_rank + length - 1:
                yield collections.Counter(range(start_rank, start_rank + length))
    
    # 连对 (>=3对)
    pair_ranks = sorted([r for r, c in cards.items() if c >= 2 and r < CARD_VALUE_2])
    for length in range(3, len(pair_ranks) + 1):
        for i in range(len(pair_ranks) - length + 1):
            start_rank = pair_ranks[i]
            if pair_ranks[i+length-1] == start_rank + length - 1:
                yield collections.Counter({r: 2 for r in range(start_rank, start_rank + length)})

    # 飞机 (>=2个三张)
    trio_ranks = sorted([r for r, c in cards.items() if c >= 3 and r < CARD_VALUE_2])
    for length in range(2, len(trio_ranks) + 1):
        for i in range(len(trio_ranks) - length + 1):
            start_rank = trio_ranks[i]
            if trio_ranks[i+length-1] == start_rank + length - 1:
                plane_body_ranks = list(range(start_rank, start_rank + length))
                plane_body_move = collections.Counter({r: 3 for r in plane_body_ranks})
                
                yield plane_body_move # 飞机不带翼
                
                remaining_for_kickers = cards - plane_body_move
                
                # 飞机带单翼
                single_kicker_pool = [r for r, c in remaining_for_kickers.items() for _ in range(c)]
                if len(single_kicker_pool) >= length:
                    for combo in combinations(single_kicker_pool, length):
                        yield plane_body_move + collections.Counter(combo)
                
                # 飞机带对翼
                pair_kicker_pool = [r for r, c in remaining_for_kickers.items() if c >= 2]
                if len(pair_kicker_pool) >= length:
                    for combo in combinations(pair_kicker_pool, length):
                        yield plane_body_move + collections.Counter({r: 2 for r in combo})


def generate_moves_with_mastercard(state: State) -> Tuple[State, ...]:
    """
    接受 state=(counts..., wildcard_count)，返回所有合法下一个 state。
    支持 wildcard(可替代任意牌)。"""
    base_counts = list(state[:-1])
    wc = state[-1]
    cards = {r: c for r, c in zip(ALL_RANKS, base_counts) if c > 0}
    moves = []  # 暂存 Counter，'*' 表示 wildcard 用量


    # wildcard-only moves: solo/pair/trio/bomb using only wildcards
    for k in (1,2,3,4):
        if wc >= k:
            cnt_wc = Counter({'*': k})
            moves.append(cnt_wc)


    # 1) 单/对/三/炸
    for r, c in cards.items():
        for k in (1, 2, 3, 4):
            need = max(0, k - c)
            if need <= wc:
                cnt = Counter({r: min(c, k)})
                if need:
                    cnt['*'] = need
                moves.append(cnt)

    # 2) Rocket
    if cards.get(20, 0) >= 1 and cards.get(30, 0) >= 1:
        moves.append(Counter({20:1, 30:1}))

    # 3) 三带一/三带二
    for r, c in cards.items():
        for attach in (1, 2):
            base_trip = min(c, 3)
            need_trip = 3 - base_trip
            if need_trip > wc:
                continue
            rem_wc = wc - need_trip
            rest = {rr:cc for rr,cc in cards.items() if rr!=r}
            for r2, c2 in rest.items():
                need_attach = max(0, attach - c2)
                if need_attach <= rem_wc:
                    cnt = Counter({r: base_trip, r2: min(c2, attach)})
                    total_wc = need_trip + need_attach
                    if total_wc:
                        cnt['*'] = total_wc
                    moves.append(cnt)

    # 4) 四带两单/两对
    for r, c in cards.items():
        base_quad = min(c, 4)
        need_quad = 4 - base_quad
        if need_quad > wc:
            continue
        rem_wc = wc - need_quad
        rest = {rr:cc for rr,cc in cards.items() if rr!=r}
        # 带两单
        singles = [r2 for r2,ct in rest.items() for _ in range(ct)]
        for combo in combinations(singles, 2):
            need_attach = sum(max(0, 1 - rest[x]) for x in combo)
            if need_attach <= rem_wc:
                cnt = Counter({r: base_quad})
                for x in combo:
                    cnt[x] += 1
                total_wc = need_quad + need_attach
                if total_wc:
                    cnt['*'] = total_wc
                moves.append(cnt)
        # 带两对
        pairs = [r2 for r2,ct in rest.items() if ct>=2]
        for combo in combinations(pairs, 2):
            need_attach = sum(max(0, 2 - rest[x]) for x in combo)
            if need_quad + need_attach <= wc:
                cnt = Counter({r: base_quad, combo[0]:2, combo[1]:2})
                total_wc = need_quad + need_attach
                if total_wc:
                    cnt['*'] = total_wc
                moves.append(cnt)

    # 5) 连对 (length>=3)
    for i in range(len(ALL_RANKS)):
        for j in range(i+2, len(ALL_RANKS)):
            seq = ALL_RANKS[i:j+1]
            L = len(seq)
            cnt = Counter()
            need = 0
            for r in seq:
                have = cards.get(r, 0)
                use = min(have, 2)
                cnt[r] = use
                need += (2 - use)
            if need <= wc:
                if need:
                    cnt['*'] = need
                moves.append(cnt)

    # 6) 飞机带单 (triplet seq length>=2)
    for i in range(len(ALL_RANKS)):
        for j in range(i+1, len(ALL_RANKS)):
            seq = ALL_RANKS[i:j+1]
            L = len(seq)
            if L < 2:
                continue
            cnt_trip = Counter()
            need_trip = 0
            for r in seq:
                have = cards.get(r, 0)
                use = min(have, 3)
                cnt_trip[r] = use
                need_trip += (3 - use)
            if need_trip > wc:
                continue
            rem_wc = wc - need_trip
            # collect single candidates
            singles = []
            for r2, c2 in cards.items():
                if r2 not in seq and c2 > 0:
                    singles += [r2]*c2
            from itertools import combinations as comb2
            for attach in comb2(singles, L):
                need_attach = L - len(attach)
                if need_attach <= rem_wc:
                    cnt = cnt_trip.copy()
                    for x in attach:
                        cnt[x] += 1
                    total_wc = need_trip + need_attach
                    if total_wc:
                        cnt['*'] = total_wc
                    moves.append(cnt)

    # 7) 飞机带两对 (triplet seq length>=2)
    for i in range(len(ALL_RANKS)):
        for j in range(i+1, len(ALL_RANKS)):
            seq = ALL_RANKS[i:j+1]
            L = len(seq)
            if L < 2:
                continue
            cnt_trip = Counter()
            need_trip = 0
            for r in seq:
                have = cards.get(r, 0)
                use = min(have, 3)
                cnt_trip[r] = use
                need_trip += (3 - use)
            if need_trip > wc:
                continue
            rem_wc = wc - need_trip
            pairs = [r2 for r2,c2 in cards.items() if r2 not in seq and c2>=2]
            for attach_pairs in combinations(pairs, L):
                need_attach = sum(max(0, 2 - cards.get(r2,0)) for r2 in attach_pairs)
                if need_attach <= rem_wc:
                    cnt = cnt_trip.copy()
                    for x in attach_pairs:
                        cnt[x] += 2
                    total_wc = need_trip + need_attach
                    if total_wc:
                        cnt['*'] = total_wc
                    moves.append(cnt)

def dfs_search(cards: CardCounter, memo: Memo) -> int:
    cards_tuple = tuple(sorted(cards.items()))
    if not cards_tuple:
        return 0
    if cards_tuple in memo:
        return memo[cards_tuple]

    min_steps = sum(cards.values())

    for move in _generate_moves(cards):
        if all(cards.get(r, 0) >= c for r, c in move.items()):
            remaining_cards = cards - move
            result = 1 + dfs_search(remaining_cards, memo)
            min_steps = min(min_steps, result)

    memo[cards_tuple] = min_steps
    return min_steps

def get_min_steps_to_win(hand_cards: List[int]) -> int:
    memo: Memo = {}
    card_counter = collections.Counter(hand_cards)
    return dfs_search(card_counter, memo)


if __name__ == '__main__':
    start_time = time.time()
    # 测试用例1: 连对 + 飞机
    hand1 = [3, 3, 4, 4, 5, 5, 7, 7, 7, 8, 8, 8, 9, 10]
    steps1 = get_min_steps_to_win(hand1)
    print(f"手牌 {hand1}")
    print(f"最少步数是: {steps1} (预期: 2, 即 334455 + 777888带9,10)")

    # 测试用例2: 飞机带对子
    hand2 = [6, 6, 6, 7, 7, 7, 9, 9, 10, 10]
    steps2 = get_min_steps_to_win(hand2)
    print(f"\n手牌 {hand2}")
    print(f"最少步数是: {steps2} (预期: 1, 即 666777带99)")

    # 测试用例3: 您的第一个用例
    hand3 = ['3', '3', '3', '3', '4', '6', '8', '8']
    # 转换为整数表示
    hand3_int = [3, 3, 3, 3, 4, 6, 8, 8]
    steps3 = get_min_steps_to_win(hand3_int)
    print(f"\n手牌 {hand3_int}")
    print(f"最少步数是: {steps3} (预期: 2)")

     # 测试用例3: 您的第一个用例
    
    hand3_int = [3, 3, 3, 4, 4, 5,5,6,6, 7, 7]
    steps3 = get_min_steps_to_win(hand3_int)
    print(f"\n手牌 {hand3_int}")
    print(f"最少步数是: {steps3} (预期: 3)")

    enumerate_time = time.time() - start_time
    print(f"枚举所有可能的出牌组合耗时: {enumerate_time:.4f} 秒")