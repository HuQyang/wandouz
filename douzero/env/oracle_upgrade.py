import collections
from collections import deque,Counter
from functools import lru_cache
from itertools import combinations
from typing import Tuple, Iterator,List
import time
import itertools



# 所有可能的牌面 value
ALL_RANKS = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,30]

State = Tuple[int,...]  # 状态表示为 len(ALL_RANKS) 的 count 元组

def hand_to_state(hand: list[int]) -> State:
    cnt = collections.Counter(hand)
    return tuple(cnt[r] for r in ALL_RANKS)

# 动作也用同样的 tuple 表示
@lru_cache(maxsize=None)
def generate_moves(state: State) -> Tuple[State,...]:
    """
    输入 state(tuple counts)，返回所有 legal move 的 tuple 列表。
    """
    cards = {r: c for r,c in zip(ALL_RANKS, state) if c>0}
    moves = []
    # 1) 单张／对子／三张／炸弹
    for r,c in cards.items():
        for k in (1,2,3,4):
            if c>=k:
                m = collections.Counter({r:k})
                moves.append(m)
    # 2) 王炸
    if cards.get(20,0) and cards.get(30,0):
        moves.append(collections.Counter({20:1,30:1}))
    # 3) 三带一／三带二
    for r,c in cards.items():
        if c>=3:
            rest = cards.copy(); rest[r]-=3
            for r2,c2 in rest.items():
                if c2>=1:
                    moves.append(collections.Counter({r:3, r2:1}))
                if c2>=2:
                    moves.append(collections.Counter({r:3, r2:2}))
    # 4) 四带两单／四带两对 
    for r, c in cards.items():
        if c >= 4:
            # 四带两单：从剩余所有单牌中任取2张
            rest_singles = [x for x,ct in cards.items() if x!=r for _ in range(ct)]
            for attach in combinations(rest_singles, 2):
                m = Counter({r:4})
                m.update(attach)
                moves.append(m)
            # 四带两对：从剩余所有对中任取2对
            rest_pairs = [x for x,ct in cards.items() if x!=r and ct>=2]
            for pair_combo in combinations(rest_pairs, 2):
                m = Counter({r:4})
                m.update({pair_combo[0]:2, pair_combo[1]:2})
                moves.append(m)
    # 5a) 连对（长度 ≥ 3）
    # 找所有 count>=2 的连续区间
    pair_ranks = [r for r,c in cards.items() if c >= 2]
    idxs = [ALL_RANKS.index(r) for r in pair_ranks]
    for start in range(len(ALL_RANKS)):
        for end in range(start+2, len(ALL_RANKS)):
            seq = ALL_RANKS[start:end+1]
            if all(cards.get(r,0) >= 2 for r in seq):
                m = Counter({r: 2 for r in seq})
                moves.append(m)

    # 5b) 飞机带单（triplet sequence 长度 ≥ 2，且每个三带一个单）
    trip_ranks = [r for r,c in cards.items() if c >= 3]
    for start in range(len(ALL_RANKS)):
        for end in range(start+1, len(ALL_RANKS)):
            seq = ALL_RANKS[start:end+1]
            L = len(seq)
            if L >= 2 and all(cards.get(r,0) >= 3 for r in seq):
                # 可选的单牌来源（不包含 seq 中的牌）
                singles = []
                for r,c in cards.items():
                    if r not in seq and c >= 1:
                        singles += [r] * c
                # 从所有单牌中选 L 张不同位置（允许同点数多张时重数）作为带单
                for attach in combinations(singles, L):
                    m = Counter({r: 3 for r in seq})
                    m.update(attach)
                    moves.append(m)
    # 6) 飞机带两对翅膀（triplet sequence + 对子）  
    #    对长度 ≥ 2 的连续三张序列，每个三带一个对子
    trip_ranks = [r for r,c in cards.items() if c >= 3]
    # 找所有可能的连续区间 [i:j]
    idxs = {r:i for i,r in enumerate(ALL_RANKS)}
    for i in range(len(ALL_RANKS)):
        for j in range(i+1, len(ALL_RANKS)):
            seq = ALL_RANKS[i:j+1]
            L = len(seq)
            if L >= 2 and all(cards.get(r,0) >= 3 for r in seq):
                # wings 对子来源：在 seq 外 ct>=2 的点数
                wing_pairs = [r for r in ALL_RANKS if r not in seq and cards.get(r,0) >= 2]
                if len(wing_pairs) >= L:
                    for attach_pairs in combinations(wing_pairs, L):
                        m = Counter({r:3 for r in seq})
                        for w in attach_pairs:
                            m[w] = 2
                        moves.append(m)


    # 最后把 Counter 转成 tuple
    uniq = []
    for m in moves:
        tup = tuple(m.get(r,0) for r in ALL_RANKS)
        if any(tup) and tup not in uniq:
            uniq.append(tup)
    return tuple(uniq)



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

    # 转换为 state tuples
    uniq = set()
    results = []
    for cnt in moves:
        used = cnt.get('*', 0)
        new_counts = []
        for orig, r in zip(base_counts, ALL_RANKS):
            use = cnt.get(r, 0)
            new_counts.append(orig - min(orig, use))
        new_wc = wc - used
        if new_wc < 0:
            continue
        ns = tuple(new_counts) + (new_wc,)
        if ns not in uniq:
            uniq.add(ns)
            results.append(ns)
    return tuple(results)

def _state_key(hand: List[int], wildcard: int) -> State:
        cnt = Counter(hand)
        return tuple(cnt[r] for r in ALL_RANKS) + (wildcard,)

# def get_min_steps_to_win_bfs(handcards: list[int],
#                             wildcards: list[int],
#                             exact: bool = True) -> int:
#     """BFS 找到最短出完手牌的步数。"""
#     wildcard_count = len(wildcards)
#     # start = _state_key(hand, wildcards)
#     # start = hand_to_state(hand)
#     # if sum(start)==0:
#     #     return 0

#     cnt = Counter(handcards)
#     start = tuple(cnt[r] for r in ALL_RANKS) + (wildcard_count,)
    
#     if sum(start) == 0:
#         return 0
#     seen = {start}

#     seen = {start}
#     q = deque([(start, 0)])
#     # while q:
#     #     state, steps = q.popleft()
#     #     for mv in generate_moves_with_mastercard(state):
#     #         # subtract move
#     #         next_state = tuple(s - m for s,m in zip(state, mv))
#     #         if any(x<0 for x in next_state):
#     #             continue
#     #         if sum(next_state)==0:
#     #             return steps + 1
#     #         if next_state not in seen:
#     #             seen.add(next_state)
#     #             q.append((next_state, steps+1))
#     # return -1  # 理论不会到这里
#     # return float('inf')
#     while q:
#         state, steps = q.popleft()
#         for ns in generate_moves_with_mastercard(state):
#             if ns not in seen:
#                 if sum(ns[:-1]) == 0 and ns[-1] == 0:
#                     return steps + 1
#                 seen.add(ns)
#                 q.append((ns, steps + 1))
#     # 理论上不会到这里
#     return -1

def get_min_steps_to_win_bfs(handcards: List[int], wildcards: List[int]) -> int:
    """
    计算最小步数，handcards: 实际手牌列表，不包括 wildcards。
    wildcards: wildcard 列表，用于计数替代。
    """
    wildcard_count = len(wildcards)
    cnt = Counter(handcards)
    start = tuple(cnt[r] for r in ALL_RANKS) + (wildcard_count,)
    if sum(start) == 0:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        state, steps = q.popleft()
        for ns in generate_moves_with_mastercard(state):
            if ns not in seen:
                if sum(ns) == 0:
                    return steps + 1
                seen.add(ns)
                q.append((ns, steps + 1))
    return float('inf')



# # 测试
# if __name__=='__main__':
#     for hand in [[3,3,3,3,4,6,8,8], [3,3,3,4,4,5,5,6,6,7,7]]:
#         print(hand, '→', get_min_steps_to_win_bfs(hand))


if __name__ == '__main__':
    # 测试用例1: 连对 + 飞机
    start_time = time.time()
    mastercard = [ 3,4 ]

    # 测试用例2: 飞机带对子
    hand2 = [6, 6, 6, 7, 7, 7, 9, 9, 10, 10]
    # hand2.extend(mastercard)
    steps2 = get_min_steps_to_win_bfs(hand2, mastercard)
    print(f"\n手牌 {hand2}")
    print(f"最少步数是: {steps2} (预期: 2, 即 666777带991010 +34)")


    hand3 = [3,3, 4,5, 7, 7, 7, 9, 9,12]
    steps3 = get_min_steps_to_win_bfs(hand3,mastercard)
    print(f"\n手牌 {hand3}")
    print(f"最少步数是: {steps3} (预期: 2)")

     # 测试用例3: 您的第一个用例
    
    hand3_int = [3, 3, 3, 4, 4, 5,5,6,6, 7, 7]
    steps3 = get_min_steps_to_win_bfs(hand3_int,mastercard)
    print(f"\n手牌 {hand3_int}")
    print(f"最少步数是: {steps3} (预期: 2)")

    hand3_int = [3,  4, 4, 5,5,6,6, 7, 7]
    steps3 = get_min_steps_to_win_bfs(hand3_int,mastercard)
    print(f"\n手牌 {hand3_int}")
    print(f"最少步数是: {steps3} (预期: 2)")

    import random
    for i in range(100):
        hand = random.choices(ALL_RANKS, k=random.randint(5, 17))
        # hand.extend(mastercard)
        steps = get_min_steps_to_win_bfs(hand.sort(),mastercard)
        print(f"手牌 {hand} → 最少步数: {steps}")

    enumerate_time = time.time() - start_time
    print(f"枚举所有可能的出牌组合耗时: {enumerate_time :.4f} 秒")
