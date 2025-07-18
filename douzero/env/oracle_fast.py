import collections
import heapq
from collections import Counter
from itertools import combinations
from typing import Tuple, List

# 定义状态和牌面顺序
ALL_RANKS = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,30]
State = Tuple[int, ...]  # 最后一个元素为 wildcard 数量


def generate_moves_with_mastercard(state: State) -> Tuple[State, ...]:
    """
    输入 state: 各 rank 的 counts + wildcard count，返回所有 legal next states.
    支持 wildcard( master card ) 补齐缺失牌。
    """
    base_counts = list(state[:-1])
    wc = state[-1]
    # 当前手牌映射
    cards = {r: c for r, c in zip(ALL_RANKS, base_counts) if c > 0}
    moves = []  # 暂存 Counter 动作，键 '*' 表示 wildcard 使用数量

    def add_move(cnt: Counter):
        moves.append(cnt)

    # 1) 单/对/三/炸
    for r, c in cards.items():
        for k in (1,2,3,4):
            need = max(0, k - c)
            if need <= wc:
                cnt = Counter({r: min(c, k)})
                if need:
                    cnt['*'] = need
                add_move(cnt)

    # 2) Rocket (王炸)
    if cards.get(20,0)>=1 and cards.get(30,0)>=1:
        add_move(Counter({20:1,30:1}))

    # 3) 三带一/二
    for r, c in cards.items():
        for attach in (1,2):
            base_trip = min(c,3)
            need_trip = 3 - base_trip
            if need_trip > wc:
                continue
            rem_wc = wc - need_trip
            rest = {rr:cc for rr,cc in cards.items() if rr!=r}
            for r2,c2 in rest.items():
                need_attach = max(0, attach - c2)
                if need_attach <= rem_wc:
                    cnt = Counter({r: base_trip, r2: min(c2, attach)})
                    total_wc = need_trip + need_attach
                    if total_wc:
                        cnt['*'] = total_wc
                    add_move(cnt)

    # 4) 四带两单/两对
    for r, c in cards.items():
        base_quad = min(c,4)
        need_quad = 4 - base_quad
        if need_quad > wc:
            continue
        rem_wc = wc - need_quad
        rest = {rr:cc for rr,cc in cards.items() if rr!=r}
        # 带两单
        singles = [r2 for r2,ct in rest.items() for _ in range(ct)]
        for combo in combinations(singles, 2):
            need_attach = sum(max(0,1 - rest[x]) for x in combo)
            if need_attach <= rem_wc:
                cnt = Counter({r: base_quad})
                for x in combo:
                    cnt[x] += 1
                total_wc = need_quad + need_attach
                if total_wc:
                    cnt['*'] = total_wc
                add_move(cnt)
        # 带两对
        pairs = [r2 for r2,ct in rest.items() if ct>=2]
        for combo in combinations(pairs,2):
            need_attach = sum(max(0,2 - rest[x]) for x in combo)
            if need_quad + need_attach <= wc:
                cnt = Counter({r: base_quad, combo[0]:2, combo[1]:2})
                total_wc = need_quad + need_attach
                if total_wc:
                    cnt['*'] = total_wc
                add_move(cnt)

        # 5) 连对 (sequence of pairs, length ≥ 3)
    for i in range(len(ALL_RANKS)):
        for j in range(i+2, len(ALL_RANKS)):
            seq = ALL_RANKS[i:j+1]
            L = len(seq)
            if L < 3:
                continue
            # 计算需要的 wildcard 数量
            need = 0
            cnt = Counter()
            for r in seq:
                have = cards.get(r, 0)
                use = min(have, 2)
                cnt[r] = use
                need += (2 - use)
            if need <= wc:
                if need > 0:
                    cnt['*'] = need
                add_move(cnt)

    # 6) 飞机带单 (triplet sequence length ≥ 2, each with one single)
    for i in range(len(ALL_RANKS)):
        for j in range(i+1, len(ALL_RANKS)):
            seq = ALL_RANKS[i:j+1]
            L = len(seq)
            if L < 2:
                continue
            # 基础三张与 wildcard
            need_trip = 0
            cnt_trip = Counter()
            for r in seq:
                have = cards.get(r, 0)
                use = min(have, 3)
                cnt_trip[r] = use
                need_trip += (3 - use)
            if need_trip > wc:
                continue
            rem_wc = wc - need_trip
            # 附带的单张来源
            singles = []
            for r2, c2 in cards.items():
                if r2 not in seq and c2 > 0:
                    singles += [r2] * c2
            # 选择 L 张单牌
            for attach in combinations(singles, min(L, len(singles))):
                need_attach = L - len(attach)
                if need_attach <= rem_wc:
                    cnt = cnt_trip.copy()
                    for r2 in attach:
                        cnt[r2] += 1
                    total_wc = need_trip + need_attach
                    if total_wc > 0:
                        cnt['*'] = total_wc
                    add_move(cnt)

    # 7) 飞机带二对 (triplet sequence length ≥ 2, each with one pair)
    for i in range(len(ALL_RANKS)):
        for j in range(i+1, len(ALL_RANKS)):
            seq = ALL_RANKS[i:j+1]
            L = len(seq)
            if L < 2:
                continue
            # 基础三张与 wildcard
            need_trip = 0
            cnt_trip = Counter()
            for r in seq:
                have = cards.get(r, 0)
                use = min(have, 3)
                cnt_trip[r] = use
                need_trip += (3 - use)
            if need_trip > wc:
                continue
            rem_wc = wc - need_trip
            # 附带的对子来源点数
            pairs = [r2 for r2, c2 in cards.items() if r2 not in seq and c2 >= 2]
            for attach_pairs in combinations(pairs, min(L, len(pairs))):
                need_attach = L*2 - sum(min(cards.get(r2,0), 2) for r2 in attach_pairs)
                if need_attach <= rem_wc:
                    cnt = cnt_trip.copy()
                    for r2 in attach_pairs:
                        cnt[r2] += 2
                    total_wc = need_trip + need_attach
                    if total_wc > 0:
                        cnt['*'] = total_wc
                    add_move(cnt)

    # 转化为后继状态 tuple
    seen = set()
    results = []
    for cnt in moves:
        used_wc = cnt.get('*',0)
        new_counts = []
        for orig, r in zip(base_counts, ALL_RANKS):
            use = cnt.get(r,0)
            new_counts.append(orig - min(orig, use))
        new_wc = wc - used_wc
        if new_wc < 0:
            continue
        ns = tuple(new_counts) + (new_wc,)
        if ns not in seen:
            seen.add(ns)
            results.append(ns)
    return tuple(results)


class OracleMinSteps:
    def __init__(self):
        self._cache_steps: dict[State,int] = {}
        self._max_take: int | None = None

    def _state_key(self, hand: List[int], wildcard: int) -> State:
        cnt = Counter(hand)
        return tuple(cnt[r] for r in ALL_RANKS) + (wildcard,)

    def _heuristic(self, state: State) -> int:
        rem = sum(state)
        # 预估每步至少去掉 max_take 张
        return (rem + self._max_take - 1) // self._max_take

    def _astar(self, start: State) -> int:
        if start in self._cache_steps:
            return self._cache_steps[start]
        if sum(start) == 0:
            self._cache_steps[start] = 0
            return 0
        # 预计算 max_take (首次调用)
        if self._max_take is None:
            succ = generate_moves_with_mastercard(start)
            self._max_take = max(sum(start) - sum(ns) for ns in succ) or 1
        seen_cost = {start: 0}
        pq: List[tuple[int,int,State]] = []
        heapq.heappush(pq, (self._heuristic(start), 0, start))
        while pq:
            f, g, s = heapq.heappop(pq)
            if g > seen_cost.get(s, float('inf')):
                continue
            if sum(s) == 0:
                self._cache_steps[start] = g
                return g
            for ns in generate_moves_with_mastercard(s):
                ng = g + 1
                if ng < seen_cost.get(ns, float('inf')):
                    seen_cost[ns] = ng
                    heapq.heappush(pq, (ng + self._heuristic(ns), ng, ns))
        self._cache_steps[start] = float('inf')
        return float('inf')

    def get_min_steps(self, hand: List[int], wildcard: int, exact: bool=True) -> int:
        """
        hand: 当前手牌列表
        wildcard: wildcard 数量
        exact: True 用 A* 精确，False 用贪心近似
        """
        start = self._state_key(hand, wildcard)
        if exact:
            return self._astar(start)
        # 近似：每步删除最多剩余最少的状态
        steps = 0
        s = start
        while sum(s) > 0:
            succ = generate_moves_with_mastercard(s)
            s = min(succ, key=lambda x: sum(x))
            steps += 1
        return steps

# 全局接口
_oracle = OracleMinSteps()

def get_min_steps_to_win_bfs(handcards: list[int],
                            wildcards: list[int],
                            exact: bool = True) -> int:
    """
    handcards: 只含正常牌的列表
    wildcards: wildcard 列表（比如 [3,4] 表示两张可替代任意牌）
    exact:     True 用 A*，False 用贪心近似
    """
    wildcard_count = len(wildcards)
    return _oracle.get_min_steps(handcards, wildcard_count, exact)



if __name__ == '__main__':
    # 测试用例1: 连对 + 飞机
    import time
    start_time = time.time()
    mastercard = [3,4]

    # 测试用例2: 飞机带对子
    hand2 = [6, 6, 6, 7, 7, 7, 9, 9, 10, 10]
    hand2.extend(mastercard)
    steps2 = get_min_steps_to_win_bfs(hand2, mastercard)
    print(f"\n手牌 {hand2}")
    print(f"最少步数是: {steps2} (预期: 2, 即 666777带991010 +34)")


    hand3 = [3, 3, 3, 3, 4, 6, 8, 8]
    steps3 = get_min_steps_to_win_bfs(hand3,mastercard)
    print(f"\n手牌 {hand3}")
    print(f"最少步数是: {steps3} (预期: 2)")

     # 测试用例3: 您的第一个用例
    
    hand3_int = [3, 3, 3, 4, 4, 5,5,6,6, 7, 7]
    steps3 = get_min_steps_to_win_bfs(hand3_int,mastercard)
    print(f"\n手牌 {hand3_int}")
    print(f"最少步数是: {steps3} (预期: 2)")

    import random
    for i in range(100):
        hand = random.choices(ALL_RANKS, k=random.randint(5, 17))
        steps = get_min_steps_to_win_bfs(hand,mastercard)
        print(f"手牌 {hand} → 最少步数: {steps}")

    enumerate_time = time.time() - start_time
    print(f"枚举所有可能的出牌组合耗时: {enumerate_time :.4f} 秒")
