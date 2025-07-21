#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from douzero.env.game import GameEnv
from douzero.env.env import is_mastercard
from douzero.env.move_generator import MovesGener
from douzero.evaluation.deep_agent import DeepAgent, DeepAgent_Alpha
from douzero.evaluation.random_agent import RandomAgent
import time 
import torch

# === 用户输入牌转换 ===
def parse_card_input(card_str):
    name2val = {
        '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
        '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
        'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30
    }
    try:
        return [name2val[c.strip().upper()] for c in card_str.strip().split()]
    except KeyError as e:
        print(f"[输入错误] 不支持的牌名: {e}")
        return None

# === 牌转换为字符串 ===
def convert_to_readable_cards(cards, mastercards=None):
    val2name = {
        11: "J", 12: "Q", 13: "K", 14: "A", 17: "2", 20: "X", 30: "D"
    }
    readable = []
    for c in cards:
        name = val2name.get(c, str(c))
        if mastercards and c in mastercards:
            name += "*"
        readable.append(name)
    return readable

# === 出牌识别 ===
def identify_move_type(move, hand_cards, mastercards):
    # md = MovesGener(hand_cards, mastercards)
    import douzero.env.move_detector as md
    move_type = md.get_move_type(move, mastercards)
    return move_type

# === 牌型转换 ===
def convert_move_type_to_readable(move_type):
    move_type_dict = {
        1: '单张',
        2: '对子',
        3: '三张',
        4: '炸弹',
        5: '王炸',
        6: '三带一',
        7: '三带二',
        8:'顺子',
        9:'对子顺子',
        10:'三张顺子',
        11:'顺子三带一',
        12:'顺子三带二',
        13:'四带二',
        14:'四带二加二',        
    }
    return move_type_dict.get(move_type, 0)

# === 加载模型 ===
def load_models(card_play_model_path_dict,mode='eval'):
    players = {}
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        model_path = card_play_model_path_dict[position]
        if model_path == 'random':
            players[position] = RandomAgent()
        else:
            # print("card_play_model_path_dict",model_path)
            # if 'weights' in model_path:
            #     players[position] = DeepAgent(position, model_path)
            # else:
            #     print("model_path", model_path)
            #     players[position] = DeepAgent_Alpha(position, model_path)
            players[position] = DeepAgent(position, model_path,mode)
    return players

# === 自动对局函数 ===
def play_auto_game(card_play_data, card_play_model_path_dict, verbose=True,show_action=False,mode='eval'):
    players = load_models(card_play_model_path_dict,mode=mode)
    if 'weight' in card_play_model_path_dict:
        import douzero.env.game as game
        env = game.GameEnv(players, show_action=show_action)
    else:
        import douzero.env.game_alpha as game_alpha
        env = game_alpha.GameEnv(players, show_action=show_action)
    # env = GameEnv(players, show_action=False, mode='eval')
    env.card_play_init(card_play_data)
    done = False
    mastercards = card_play_data['mastercard_list']
    obs = env.get_infoset()

    if verbose:
        print("========== 初始牌局：==========")
        for k, v in card_play_data.items():
            print(k, " ".join(convert_to_readable_cards(v)))
        print("==============================")

    while not done:
        player = env.acting_player_position
        hand_cards = env.info_sets[player].player_hand_cards
        legal_actions = env.get_infoset().legal_actions
        action = players[player].act(obs)
        if verbose:
            print(f"\n=== 出牌方: {player} ===")
            print("手牌:", " ".join(convert_to_readable_cards(sorted(hand_cards), mastercards)))
            if action != 'pass':
                print("出牌:", " ".join(convert_to_readable_cards(action, mastercards)))
                if any(is_mastercard(c) for c in action):
                    print("使用了癞子牌！")
                move_type = identify_move_type(action, hand_cards, mastercards)
                # print(f"出牌类型: {convert_move_type_to_readable(move_type['type'])}")
            else:
                print("选择不出（pass）")
        obs, _, done, _ = env.step(action)
    if verbose:
        print("\n 游戏结束！")

# === 主函数 ===
def play_interactive_game(card_play_data, card_play_model_path_dict, human_role='landlord_up',show_card=True):
    players = load_models(card_play_model_path_dict,mode='train')

    print("========== 初始牌局：==========")
    for k,v in card_play_data.items():
        print(k," ".join(convert_to_readable_cards(v)))
    print("==============================")
    env = GameEnv(players,show_action=False,mode='eval')
    env.card_play_init(card_play_data)  # 先初始化卡牌
    done = False
    mastercards = card_play_data['mastercard_list']
    obs = env.get_infoset()

    while not done:
        player = env.acting_player_position
        hand_cards = env.info_sets[player].player_hand_cards

        print(f"\n=== 出牌方: {player} ===")
        if show_card:
            print("手牌:", " ".join(convert_to_readable_cards(sorted(hand_cards), mastercards)))
        else:
            if player == human_role:
                print("手牌:", " ".join(convert_to_readable_cards(sorted(hand_cards), mastercards)))
            else:
                print("手牌数：", len(hand_cards))
        if player == human_role:
            # 用户输入
            while True:
                legal_actions = env.get_infoset().legal_actions
                # print("length of legal actions:", len(legal_actions))
                # print("legal actions:", legal_actions)
                user_input = input("请输入要出的牌，或pass（输入空格后回车）：")
                if user_input.lower().strip() == 'pass':
                    action = []
                    break
                action = parse_card_input(user_input)
                if action is None:
                    continue    
                if not all(c in hand_cards for c in action):
                    print("[输入错误] 你出的牌不在你的手牌中，请重新输入！")
                    continue
                
                # 新加的：检查是不是合法动作！
                legal_actions = env.get_infoset().legal_actions
                # if action not in legal_actions:
                if not any(sorted(action) == sorted(legal_action) for legal_action in legal_actions):
                    print("legal_actions:", legal_actions)
                    print("[输入错误] 你的出牌不符合规则（非法出牌类型），请重新输入！")
                    # print(f"提示：你当前可以出的牌有 {len(legal_actions)} 种。可以输入 'tips' 查看。")
                    continue
                else:
                    break
        else:
            # 模型行动
            # action = players[player].act(obs)
            legal_actions = env.get_infoset().legal_actions
            # print("length of legal actions:", len(legal_actions))
            # print("legal actions:", legal_actions)
            start_time = time.time()
            action = players[player].act(obs)
            duration = (time.time() - start_time)*1000
            # print(f"[模型响应时间] {duration:.4f} 毫秒 ")

        # 输出结果
        if action != 'pass':
            print("出牌:", " ".join(convert_to_readable_cards(action, mastercards)))
            if any(is_mastercard(c) for c in action):
                print("使用了癞子牌！")
            move_type = identify_move_type(action, hand_cards, mastercards)
            print(f"出牌类型: {convert_move_type_to_readable(move_type['type'])}")
        else:
            print("选择不出（pass）")

        # 执行出牌
        obs, _, done, _ = env.step(action)

    print("\n 游戏结束！")

# === 示例入口 ===
if __name__ == '__main__':
    # 示例对局数据
    from generate_eval_data import generate
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # card_play_data = generate()

    # card_play_data = {
    #     'landlord':      [3, 3, 3, 5, 5, 6, 7, 7, 7, 7, 9, 10, 14, 14, 14, 17, 20, 6, 8, 10],
    #     'landlord_up':   [3, 4, 4, 4, 6, 6, 8, 9, 10, 10, 11, 12, 12, 12, 13, 17, 30],
    #     'landlord_down': [4, 5, 5, 6, 8, 8, 9, 9, 11, 11, 12, 13, 13, 13, 14, 17, 17],
    #     'mastercard_list': [4, 8],  # 癞子点数是 4 和 8
    #     'three_landlord_cards': [6,8,10]  # 
    # }

    # card_play_data = {
    #     'landlord':      [3, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10, 11, 12, 12, 17, 17],
    #     'landlord_up':   [3, 3, 4, 4, 4, 7, 8, 9, 9, 10, 11, 13, 13, 14, 14, 17, 20],
    #     'landlord_down': [3, 4, 6, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 17, 30],
    #     'mastercard_list': [5, 11],  # 癞子点数是 4 和 8
    #     'three_landlord_cards': [3, 5, 5]  # 
    # }

    # card_play_data = {
    #     'landlord':      [3, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 17, 17, 20],
    #     'landlord_up':   [3, 4, 5, 6, 7, 7, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 14],
    #     'landlord_down': [3, 3, 4, 4, 5, 6, 6, 8, 8, 9, 11, 11, 13, 14, 17, 17, 30],
    #     'mastercard_list': [9, 13], 
    #     'three_landlord_cards': [7, 14, 17]   
    # }

    # card_play_data = {
    #     'landlord':      [3, 3, 3, 3, 5,5, 7, 9, 9, 9, 10, 10, 11, 12, 12, 12, 12, 14, 14, 20],
    #     'landlord_up':   [4, 4, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 10, 10, 11, 13],
    #     'landlord_down': [4, 4, 5, 7, 9, 11, 11, 13, 13, 13, 14, 14, 17, 17, 17, 17, 30],
    #     'mastercard_list': [10, 17], 
    #     'three_landlord_cards': [5,5,20]   
    # }

    # card_play_data = {
    #     'landlord':      [3, 3, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 11, 11, 11, 12, 13, 17],
    #     'landlord_up':   [3, 4, 4, 5, 5, 5, 6, 7, 9, 10, 10, 11, 13, 13, 14, 17, 17],
    #     'landlord_down': [3, 4, 4, 7, 8, 9, 10, 12, 12, 12, 13, 14, 14, 14, 17, 20, 30],
    #     'mastercard_list': [11, 13], 
    #     'three_landlord_cards': [6, 11, 12]  
    # }

    # card_play_data = {
    #     'landlord':     [3,3,4,4,5,5,6,7,7,7,9,11,11,12,12,14,14,17,17,20],
    #     'landlord_up':   [3,5,5,6,8,8,8,9,9,10,12,13,13,13,14,14,17],
    #     'landlord_down': [3,4,4,6,6,7,8,9,10,10,10,11,11,12,13,17,30],
    #     'mastercard_list': [9,11],  
    #     'three_landlord_cards': [4, 7, 14]
    # }

    # card_play_data = {
    #     'landlord':     [4,4,5,5,5,5,6,6,6,6,7,9,10,10,10,10,13,14,14,30],
    #     'landlord_up':   [3,3,3,3,4,7,8,8,8,8,9,12,12,12,12,13,20],
    #     'landlord_down': [4,7,7,9,9,11,11,11,11,13,13,14,14,17,17,17,17],
    #     'mastercard_list': [ ],  
    #     'three_landlord_cards': [4, 4, 13]
    # }

    # card_play_data = {
    #     'landlord':     [3, 4, 4, 5, 5, 5, 6, 6, 8, 8, 8, 9, 10, 10, 13, 14, 14, 14, 20, 30],
    #     'landlord_up':   [3, 4, 5, 7, 7, 9, 9, 9, 10, 11, 11, 11, 13, 14, 17, 17, 17],
    #     'landlord_down': [3, 3, 4, 6, 6, 7, 7, 8, 10, 11, 12, 12, 12, 12, 13, 13, 17],
    #     'mastercard_list': [4, 9],  
    #     'three_landlord_cards': [4, 5, 6]
    # }

    # card_play_data = {
    #     'landlord':     [3, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 12, 13, 14, 17, 17, 20, 30],
    #     'landlord_up':   [3, 4, 5 ,6, 7, 8,5, 6, 7, 8, 9,10, 13,14, 14,17, 17 ],
    #     'landlord_down': [3, 3, 4, 5, 6, 7, 9, 9, 10, 10,11, 11, 12, 12, 13, 13, 14],
    #     'mastercard_list': [ ],  
    #     'three_landlord_cards': [4, 11, 12]
    # }

    card_play_data = {
        'landlord':      [3, 3, 3, 5, 5, 6, 7, 7, 7, 7, 9, 10, 14, 14, 14, 17, 20, 6, 8, 10],
        'landlord_up':   [3, 4, 4, 4, 6, 6, 8, 9, 10, 10, 11, 12, 12, 12, 13, 17, 30],
        'landlord_down': [4, 5, 5, 6, 8, 8, 9, 9, 11, 11, 12, 13, 13, 13, 14, 17, 17],
        'mastercard_list': [3,4 ],  # 癞子点数是 4 和 8
        'three_landlord_cards': [6,8,10]  # 
    }

    # card_play_data = generate()

    
    # ite = 1342233600 # 21 days 
    # ite = 504112000 # 7 days
    # ite = 773708800 # 14days
    ite = 1412819200 #23 days
    # ite = 1438089600 #23 days
    # ite = 1462518400 #23 days

    ite1 = 773708800
    ite2 = 1342233600
    ite3 = 14054400

    alpha_ite1 = 7913600
    

    landlord = f'oracle_reward/douzero/landlord{ite3}.ckpt'
    landlord_up = f'oracle_reward/douzero/landlord_up{ite3}.ckpt'
    landlord_down = f'oracle_reward/douzero/landlord_down{ite3}.ckpt'

    # landlord = f'douzero_checkpoints/douzero/landlord_weights_{ite3}.ckpt'
    # landlord_up = f'../beta/douzero_checkpoints/douzero/landlord_up_{alpha_ite1}.ckpt'
    # landlord_down = f'../beta/douzero_checkpoints/douzero/landlord_down_{alpha_ite1}.ckpt'

    # landlord = f'../beta/douzero_checkpoints/douzero/landlord_{alpha_ite1}.ckpt'
    # landlord_up = f'douzero_checkpoints/douzero/landlord_up_weights_{ite3}.ckpt'
    # landlord_down = f'douzero_checkpoints/douzero/landlord_down_weights_{ite3}.ckpt'

    # landlord = f'douzero_checkpoints/douzero/landlord_weights_{ite3}.ckpt'
    # landlord_up = f'../beta/douzero_checkpoints/douzero/landlord_up_{alpha_ite1}.ckpt'
    # landlord_down = f'../beta/douzero_checkpoints/douzero/landlord_down_{alpha_ite1}.ckpt'
    

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down
    }

    print("\n 斗地主人机对战开始！你扮演地主。")
    print(f'load ckpt {landlord}')
    # play_interactive_game(card_play_data, card_play_model_path_dict, human_role='landlord_up')
    play_auto_game(card_play_data, card_play_model_path_dict,mode='train')