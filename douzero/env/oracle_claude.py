import collections
from typing import List, Dict, Set, Tuple, Iterator
from itertools import combinations, product
import time

# --- 常量定义 ---
CARD_VALUE_2 = 17
BLACK_JOKER_VALUE = 20
RED_JOKER_VALUE = 30
ALL_RANKS = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,30]

CardCounter = Dict[int, int]
Memo = Dict[Tuple[int, ...], int]

# ===================================================================
# 高性能牌型生成器 (修复版)
# ===================================================================

def generate_all_moves_fast(cards: CardCounter) -> Iterator[CardCounter]:
    """高性能版本的牌型生成器 - 修复四带两对问题"""
    
    # --- 基础牌型 (单张、对子、三张、炸弹) ---
    for r, c in cards.items():
        if c >= 1: yield {r: 1}
        if c >= 2: yield {r: 2}
        if c >= 3: yield {r: 3}
        if c >= 4: yield {r: 4}
    
    # 王炸
    if cards.get(BLACK_JOKER_VALUE, 0) >= 1 and cards.get(RED_JOKER_VALUE, 0) >= 1:
        yield {BLACK_JOKER_VALUE: 1, RED_JOKER_VALUE: 1}

    # --- 只生成最常用的连续牌型，减少搜索空间 ---
    ranks_no_2 = sorted([r for r in cards if r < CARD_VALUE_2])
    
    # 顺子 (只生成5-8张，减少长顺子)
    for length in range(5, min(9, len(ranks_no_2) + 1)):
        for i in range(len(ranks_no_2) - length + 1):
            selected_ranks = ranks_no_2[i:i+length]
            if all(selected_ranks[j+1] - selected_ranks[j] == 1 for j in range(length-1)):
                if all(cards.get(r, 0) >= 1 for r in selected_ranks):
                    yield {r: 1 for r in selected_ranks}
    
    # 连对 (只生成3-5对，减少长连对)
    pair_ranks = [r for r in ranks_no_2 if cards.get(r, 0) >= 2]
    for length in range(3, min(6, len(pair_ranks) + 1)):
        for i in range(len(pair_ranks) - length + 1):
            selected_ranks = pair_ranks[i:i+length]
            if all(selected_ranks[j+1] - selected_ranks[j] == 1 for j in range(length-1)):
                yield {r: 2 for r in selected_ranks}
    
    # 飞机 (只生成2-4个，减少长飞机)
    trio_ranks = [r for r in ranks_no_2 if cards.get(r, 0) >= 3]
    for length in range(2, min(5, len(trio_ranks) + 1)):
        for i in range(len(trio_ranks) - length + 1):
            selected_ranks = trio_ranks[i:i+length]
            if all(selected_ranks[j+1] - selected_ranks[j] == 1 for j in range(length-1)):
                plane_body = {r: 3 for r in selected_ranks}
                yield plane_body  # 飞机不带翼
                
                # 计算剩余牌
                remaining = cards.copy()
                for r in selected_ranks:
                    remaining[r] -= 3
                    if remaining[r] == 0:
                        del remaining[r]
                
                # 飞机带单翼 (只考虑最优组合)
                single_kickers = []
                for r, c in remaining.items():
                    single_kickers.extend([r] * min(c, 2))  # 限制每种牌最多2张作为翼
                
                if len(single_kickers) >= length:
                    # 只取前几种组合，避免组合爆炸
                    for combo in list(combinations(single_kickers, length))[:3]:
                        move = plane_body.copy()
                        for kicker in combo:
                            move[kicker] = move.get(kicker, 0) + 1
                        yield move
                
                # 飞机带对翼 (只考虑有对子的牌)
                pair_kickers = [r for r, c in remaining.items() if c >= 2]
                if len(pair_kickers) >= length:
                    for combo in list(combinations(pair_kickers, length))[:2]:
                        move = plane_body.copy()
                        for kicker in combo:
                            move[kicker] = move.get(kicker, 0) + 2
                        yield move

    # --- 带牌组合 (修复版) ---
    for r_main, c_main in cards.items():
        # 三带一 和 三带一对
        if c_main >= 3:
            remaining = cards.copy()
            remaining[r_main] -= 3
            if remaining[r_main] == 0:
                del remaining[r_main]
            
            # 三带一
            for r_kicker in remaining.keys():
                if remaining[r_kicker] >= 1:
                    yield {r_main: 3, r_kicker: 1}
            
            # 三带一对
            for r_kicker, c_kicker in remaining.items():
                if c_kicker >= 2:
                    yield {r_main: 3, r_kicker: 2}
        
        # 四带二 (修复版 - 确保四带两对能正确生成)
        if c_main >= 4:
            remaining = cards.copy()
            remaining[r_main] -= 4
            if remaining[r_main] == 0:
                del remaining[r_main]
            
            # 四带两单 (限制组合数)
            single_kickers = []
            for r, c in remaining.items():
                single_kickers.extend([r] * min(c, 2))
            
            if len(single_kickers) >= 2:
                # 只取前几种组合
                for combo in list(combinations(single_kickers, 2))[:5]:
                    move = {r_main: 4}
                    for kicker in combo:
                        move[kicker] = move.get(kicker, 0) + 1
                    yield move
            
            # 四带两对 (修复：确保正确生成)
            pair_kickers = [r for r, c in remaining.items() if c >= 2]
            if len(pair_kickers) >= 2:
                # 生成所有可能的两对组合
                for combo in combinations(pair_kickers, 2):
                    move = {r_main: 4}
                    for kicker in combo:
                        move[kicker] = 2
                    yield move

# ===================================================================
# 高性能万能牌处理
# ===================================================================

def generate_mastercard_replacements_fast(cards: CardCounter, mastercards: Set[int]) -> Iterator[CardCounter]:
    """高性能万能牌替换方案生成器"""
    
    # 获取所有万能牌
    mastercard_counts = {}
    for mc in mastercards:
        if mc in cards:
            mastercard_counts[mc] = cards[mc]
    
    if not mastercard_counts:
        yield cards
        return
    
    # 1. 不使用万能牌
    yield cards
    
    # 2. 智能目标选择 - 只选择最有价值的目标
    targets = set()
    
    # 现有的非万能牌 (优先级最高)
    for r in cards:
        if r < CARD_VALUE_2 and r not in mastercards:
            targets.add(r)
    
    # 补充连续性的关键牌 (第二优先级)
    existing_ranks = sorted([r for r in cards if r < CARD_VALUE_2 and r not in mastercards])
    for i in range(len(existing_ranks) - 1):
        gap = existing_ranks[i + 1] - existing_ranks[i]
        if gap == 2:  # 只补充单个缺口
            targets.add(existing_ranks[i] + 1)
    
    # 限制目标数量，避免搜索爆炸
    targets = sorted(list(targets))[:8]  # 最多8个目标
    
    # 3. 高效替换方案生成
    generated_count = 0
    max_replacements = 20
    
    # 收集所有万能牌实例
    available_mastercards = []
    for mc, count in mastercard_counts.items():
        available_mastercards.extend([mc] * count)
    
    # 智能替换策略
    for target in targets:
        if generated_count >= max_replacements:
            break
            
        max_to_use = min(len(available_mastercards), 4)
        
        for num_to_convert in range(1, max_to_use + 1):
            if generated_count >= max_replacements:
                break
                
            if num_to_convert == 1:
                # 单个万能牌替换
                for mc in set(available_mastercards):
                    if mastercard_counts[mc] >= 1:
                        new_cards = cards.copy()
                        new_cards[mc] -= 1
                        if new_cards[mc] == 0:
                            del new_cards[mc]
                        new_cards[target] = new_cards.get(target, 0) + 1
                        yield new_cards
                        generated_count += 1
                        
            elif num_to_convert == 2:
                # 两个万能牌替换
                mastercard_types = list(mastercard_counts.keys())
                
                # 同类型万能牌
                for mc in mastercard_types:
                    if mastercard_counts[mc] >= 2:
                        new_cards = cards.copy()
                        new_cards[mc] -= 2
                        if new_cards[mc] == 0:
                            del new_cards[mc]
                        new_cards[target] = new_cards.get(target, 0) + 2
                        yield new_cards
                        generated_count += 1
            
            elif num_to_convert == 3:
                # 三个万能牌替换
                if len(available_mastercards) >= 3:
                    mastercard_types = list(mastercard_counts.keys())
                    if len(mastercard_types) >= 2:
                        # 2+1组合
                        for i, mc1 in enumerate(mastercard_types):
                            for j, mc2 in enumerate(mastercard_types):
                                if i != j and mastercard_counts[mc1] >= 2 and mastercard_counts[mc2] >= 1:
                                    new_cards = cards.copy()
                                    new_cards[mc1] -= 2
                                    new_cards[mc2] -= 1
                                    if new_cards[mc1] == 0:
                                        del new_cards[mc1]
                                    if new_cards[mc2] == 0:
                                        del new_cards[mc2]
                                    new_cards[target] = new_cards.get(target, 0) + 3
                                    yield new_cards
                                    generated_count += 1
                                    break
                            if generated_count >= max_replacements:
                                break
            
            elif num_to_convert == 4:
                # 四个万能牌替换
                total_mastercards = sum(mastercard_counts.values())
                if total_mastercards >= 4:
                    new_cards = cards.copy()
                    remaining_convert = 4
                    for mc in mastercard_counts:
                        take = min(mastercard_counts[mc], remaining_convert)
                        new_cards[mc] -= take
                        if new_cards[mc] == 0:
                            del new_cards[mc]
                        remaining_convert -= take
                        if remaining_convert == 0:
                            break
                    
                    if remaining_convert == 0:
                        new_cards[target] = new_cards.get(target, 0) + 4
                        yield new_cards
                        generated_count += 1
                
                # 不同类型万能牌组合
                if len(mastercard_types) >= 2:
                    mc1, mc2 = mastercard_types[0], mastercard_types[1]
                    if mastercard_counts[mc1] >= 1 and mastercard_counts[mc2] >= 1:
                        new_cards = cards.copy()
                        new_cards[mc1] -= 1
                        new_cards[mc2] -= 1
                        if new_cards[mc1] == 0:
                            del new_cards[mc1]
                        if new_cards[mc2] == 0:
                            del new_cards[mc2]
                        new_cards[target] = new_cards.get(target, 0) + 2
                        yield new_cards
                        generated_count += 1

# ===================================================================
# 高性能主算法
# ===================================================================

def calculate_min_steps_fast(cards: CardCounter, mastercards: Set[int] = None, memo: Memo = None, depth: int = 0) -> int:
    """高性能版本的最少出牌步数计算"""
    
    if memo is None:
        memo = {}
    
    if mastercards is None:
        mastercards = set()
    
    # 基础情况
    if not cards:
        return 0
    
    # 记忆化
    cards_key = tuple(sorted(cards.items()))
    if cards_key in memo:
        return memo[cards_key]
    
    total_cards = sum(cards.values())
    
    # 优化的剪枝
    if total_cards == 1:
        memo[cards_key] = 1
        return 1
    
    # 深度限制
    if depth > 8:
        memo[cards_key] = total_cards
        return total_cards
    
    # 早期剪枝：检查一步解
    if total_cards <= 8:  # 扩大检查范围，确保四带两对能被检测到
        replacement_count = 0
        for replacement_cards in generate_mastercard_replacements_fast(cards, mastercards):
            replacement_count += 1
            if replacement_count > 10:
                break
                
            for move in generate_all_moves_fast(replacement_cards):
                can_play = all(replacement_cards.get(r, 0) >= need for r, need in move.items())
                if can_play:
                    remaining = replacement_cards.copy()
                    for r, need in move.items():
                        remaining[r] -= need
                        if remaining[r] == 0:
                            del remaining[r]
                    if not remaining:
                        memo[cards_key] = 1
                        return 1
    
    min_steps = total_cards  # 最坏情况
    
    # 限制万能牌替换方案的尝试
    replacement_count = 0
    max_replacements = 15 if total_cards > 8 else 20
    
    for replacement_cards in generate_mastercard_replacements_fast(cards, mastercards):
        replacement_count += 1
        if replacement_count > max_replacements:
            break
        
        # 限制每个替换方案的出牌尝试
        move_count = 0
        max_moves = 25
        
        for move in generate_all_moves_fast(replacement_cards):
            move_count += 1
            if move_count > max_moves:
                break
            
            # 检查是否可以出这个牌
            can_play = all(replacement_cards.get(r, 0) >= need for r, need in move.items())
            
            if not can_play:
                continue
            
            # 计算剩余牌
            remaining = replacement_cards.copy()
            for r, need in move.items():
                remaining[r] -= need
                if remaining[r] == 0:
                    del remaining[r]
            
            # 递归计算
            result = 1 + calculate_min_steps_fast(remaining, mastercards, memo, depth + 1)
            min_steps = min(min_steps, result)
            
            # 强化早期剪枝
            if min_steps <= 1:
                memo[cards_key] = min_steps
                return min_steps
            
            if min_steps <= 2 and depth <= 3:
                memo[cards_key] = min_steps
                return min_steps
    
    memo[cards_key] = min_steps
    return min_steps

# ===================================================================
# 接口函数
# ===================================================================

def get_min_steps_to_win_with_mastercards(hand_cards: List[int], mastercards: List[int] = None) -> int:
    """计算最少出牌步数（支持万能牌）- 高性能版本"""
    if not hand_cards:
        return 0
    
    cards = {}
    for card in hand_cards:
        cards[card] = cards.get(card, 0) + 1
    
    mastercard_set = set(mastercards) 
    
    return calculate_min_steps_fast(cards, mastercard_set)

def get_min_steps_to_win(hand_cards: List[int]) -> int:
    """计算最少出牌步数（无万能牌）- 高性能版本"""
    return get_min_steps_to_win_with_mastercards(hand_cards, [])

# ===================================================================
# 测试函数 - 用于调试四带两对问题
# ===================================================================

def debug_four_with_two_pairs():
    """调试四带两对生成问题"""
    print("=== 调试四带两对问题 ===")
    
    cards = {3: 4, 4: 2, 5: 2}  # [3, 3, 3, 3, 4, 4, 5, 5]
    print(f"测试手牌: {cards}")
    
    print("\n生成的所有牌型:")
    moves = list(generate_all_moves_fast(cards))
    
    for i, move in enumerate(moves):
        if len(move) > 1:  # 只显示组合牌型
            print(f"{i+1}: {move}")
            if move == {3: 4, 4: 2, 5: 2}:
                print("   ★ 找到四带两对!")
    
    # 检查是否能一步出完
    for move in moves:
        can_play = all(cards.get(r, 0) >= need for r, need in move.items())
        if can_play:
            remaining = cards.copy()
            for r, need in move.items():
                remaining[r] -= need
                if remaining[r] == 0:
                    del remaining[r]
            if not remaining:
                print(f"\n一步出完方案: {move}")
                return True
    
    print("\n未找到一步出完方案!")
    return False

# ===================================================================
# 测试
# ===================================================================

if __name__ == '__main__':
    print("=== 修复后的斗地主算法测试 ===")
    
    # 先调试四带两对问题
    # debug_four_with_two_pairs()
    
    print("\n" + "="*50)
    
    # # 核心测试用例
    # test_cases = [
    #     ([3, 3, 3, 4, 4], [], 1, "三带一对 333+44"),
    #     ([3, 3, 3, 3, 4, 4, 5, 5], [], 1, "四带两对 3333+44+55"),  # 修复的重点测试
    #     ([5, 5, 7, 7, 9, 9], [5, 7], 1, "5,7→9组成炸弹 9999+99 (2步)"),
    #     ([8, 8, 8, 9, 9], [9], 1, "9→8组成炸弹 8888+9"),
    #     ([BLACK_JOKER_VALUE, RED_JOKER_VALUE], [], 1, "王炸"),
    #     ([3, 3], [], 1, "对子"),
    #     ([5], [], 1, "单张"),
    #     ([8, 8, 7, 3, 10, 4], [3, 10], 2, "重点测试用例"),
    # ]
    
    # print("核心测试用例:")
    # total_time = 0
    
    # for hand, masters, expected, desc in test_cases:
    #     start_time = time.time()
    #     result = get_min_steps_to_win_with_mastercards(hand, masters)
    #     end_time = time.time()
    #     test_time = (end_time - start_time) * 1000
    #     total_time += test_time
        
    #     status = "✓" if result == expected else "✗"
    #     print(f"{status} {desc}: {hand}, 万能牌{masters} → {result}步 (预期: {expected}步) [{test_time:.1f}ms]")
    import random
    perf_start = time.time()
    for i in range(100):  # 从100减少到50
        hand = random.choices(ALL_RANKS, k=random.randint(8, 12))  # 减少手牌数量
        hand.sort()
        mastercards = random.sample(ALL_RANKS, k=2)
        
        test_start = time.time()
        steps = get_min_steps_to_win_with_mastercards(hand, mastercards)
        test_time = time.time() - test_start
        
        if i < 5:
            print(f"  性能测试{i+1}: {hand}, {(mastercards)}万能牌类型 → {steps}步, 耗时{test_time*1000:.1f}ms")
    perf_end = time.time()
    print(f"\n总测试耗时: {(perf_end - perf_start):.2f}秒")