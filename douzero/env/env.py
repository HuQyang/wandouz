from collections import Counter
import numpy as np

from douzero.env.game import GameEnv
from douzero.env.utils import ACTION_ENCODE_DIM, ENCODE_DIM
from douzero.env.oracle_claude import get_min_steps_to_win_with_mastercards

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

global_mastercard_values = []


class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, objective,show_action=False,use_oracle_reward=False):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective
        self.use_oracle_reward = use_oracle_reward

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            self.players[position] = DummyAgent(position)
        
        self.show_action = show_action

        # Initialize the internal environment
        self._env = GameEnv(self.players,show_action=show_action)

        self.infoset = None

    def reset(self):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        self.mastercard_list = np.random.choice(range(3, 15), 2, replace=False)
        # self.mastercard_list = []
        set_global_mastercards(self.mastercard_list)
        card_play_data = {'landlord': _deck[:20],
                          'landlord_up': _deck[20:37],
                          'landlord_down': _deck[37:54],
                          'three_landlord_cards': _deck[17:20],
                          'mastercard_list': self.mastercard_list,
                          }
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """

        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            if self.use_oracle_reward:
                # oracle dense reward at the end of game
                reward = self._get_oracle_reward_cached()
            else:
                reward = self._get_reward()
        else:
            # oracle dense reward before game ends
            reward = self._get_oracle_reward_cached() if self.use_oracle_reward else 0.0

        # print("reward: ", reward)
        obs = None if done else get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            elif self.objective == 'logadp':
                return bomb_num + 1.0
            else:
                return 1.0 
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            elif self.objective == 'logadp':
                return -bomb_num - 1.0
            else:
                return -1.0 
    

    def _get_oracle_reward(self):
        """
        计算oracle reward，返回每个时间步的reward列表
        """
        hand_seq = {
            'landlord': self._env.info_sets['landlord'].handcards_seq.copy(),
            'landlord_up': self._env.info_sets['landlord_up'].handcards_seq.copy(),
            'landlord_down': self._env.info_sets['landlord_down'].handcards_seq.copy(),
        }
        
        # 确保所有序列长度一致
        max_len = max(len(hand_seq['landlord']), len(hand_seq['landlord_up']), len(hand_seq['landlord_down']))
        
        # 如果序列长度不一致，用最后一个状态填充
        for pos in hand_seq:
            while len(hand_seq[pos]) < max_len:
                if len(hand_seq[pos]) > 0:
                    hand_seq[pos].append(hand_seq[pos][-1])
                else:
                    hand_seq[pos].append([])  # 空手牌
        
        # 计算每一步的最小获胜步数
        N_landlord = [get_min_steps_to_win_with_mastercards(h, self.mastercard_list) for h in hand_seq['landlord']]
        N_peas1 = [get_min_steps_to_win_with_mastercards(h, self.mastercard_list) for h in hand_seq['landlord_up']]
        N_peas2 = [get_min_steps_to_win_with_mastercards(h, self.mastercard_list) for h in hand_seq['landlord_down']]

        # 计算advantage序列
        Adv = [nl - min(np1, np2) for nl, np1, np2 in zip(N_landlord, N_peas1, N_peas2)]

        # 计算delta_adv作为reward
        delta_adv = [Adv[0]] + [Adv[i] - Adv[i-1] for i in range(1, len(Adv))]

        # 根据出牌方给不同系数的奖励
        rewards = []
        order = ['landlord', 'landlord_up', 'landlord_down']
        
        # 这里需要根据实际的出牌顺序来分配reward
        # 假设我们有step_positions记录每步的出牌方
        step_positions = getattr(self._env, 'step_positions', [])
        
        if len(step_positions) == 0:
            # 如果没有记录，按轮次分配
            for i, da in enumerate(delta_adv):
                pos = order[i % 3]
                coef = -1.0 if pos == 'landlord' else 0.5
                rewards.append(coef * da)
        else:
            # 根据实际出牌顺序分配
            for i, da in enumerate(delta_adv):
                if i < len(step_positions):
                    pos = step_positions[i]
                    coef = -1.0 if pos == 'landlord' else 0.5
                    rewards.append(coef * da)
                else:
                    rewards.append(0.0)
        
        print(f"Oracle rewards computed: {len(rewards)} steps, rewards: {rewards}")
        return rewards
    

    def _get_oracle_reward_cached(self):
        """
        优化版本：使用缓存避免重复计算相同手牌的oracle值
        """
        # 初始化缓存（如果还没有的话）
        if not hasattr(self, '_oracle_cache'):
            self._oracle_cache = {}
        
        hand_seq = {
            'landlord': self._env.info_sets['landlord'].handcards_seq.copy(),
            'landlord_up': self._env.info_sets['landlord_up'].handcards_seq.copy(),
            'landlord_down': self._env.info_sets['landlord_down'].handcards_seq.copy(),
        }
        
        # 确保所有序列长度一致
        max_len = max(len(hand_seq['landlord']), len(hand_seq['landlord_up']), len(hand_seq['landlord_down']))
        
        for pos in hand_seq:
            while len(hand_seq[pos]) < max_len:
                if len(hand_seq[pos]) > 0:
                    hand_seq[pos].append(hand_seq[pos][-1])
                else:
                    hand_seq[pos].append([])
        
        # 使用缓存计算最小获胜步数
        def get_cached_min_steps(handcards, position):
            # 将手牌转换为可hash的tuple作为cache key
            cache_key = (tuple(sorted(handcards)), tuple(sorted(self.mastercard_list)))
            
            if cache_key not in self._oracle_cache:
                self._oracle_cache[cache_key] = get_min_steps_to_win_with_mastercards(handcards, self.mastercard_list)
            
            return self._oracle_cache[cache_key]
        
        # 批量计算，避免重复
        N_landlord = [get_cached_min_steps(h, 'landlord') for h in hand_seq['landlord']]
        N_peas1 = [get_cached_min_steps(h, 'landlord_up') for h in hand_seq['landlord_up']]
        N_peas2 = [get_cached_min_steps(h, 'landlord_down') for h in hand_seq['landlord_down']]
        
        # 计算advantage序列
        Adv = [nl - min(np1, np2) for nl, np1, np2 in zip(N_landlord, N_peas1, N_peas2)]
        
        # 计算delta_adv作为reward
        delta_adv = [Adv[0]] + [Adv[i] - Adv[i-1] for i in range(1, len(Adv))]
        
        # 根据出牌方分配reward
        rewards = []
        step_positions = getattr(self._env, 'step_positions', [])
        order = ['landlord', 'landlord_up', 'landlord_down']
        
        if len(step_positions) == 0:
            for i, da in enumerate(delta_adv):
                pos = order[i % 3]
                coef = -1.0 if pos == 'landlord' else 0.5
                rewards.append(coef * da)
        else:
            for i, da in enumerate(delta_adv):
                if i < len(step_positions):
                    pos = step_positions[i]
                    coef = -1.0 if pos == 'landlord' else 0.5
                    rewards.append(coef * da)
                else:
                    rewards.append(0.0)
        
        return rewards


    # 优化版本2：增量计算，只计算新的步骤

    def _get_oracle_reward_incremental(self):
        """
        增量版本：只计算自上次以来的新步骤
        """
        # 初始化增量计算状态
        if not hasattr(self, '_oracle_state'):
            self._oracle_state = {
                'cache': {},
                'last_computed_step': -1,
                'cached_advantages': [],
                'cached_rewards': []
            }
        
        hand_seq = {
            'landlord': self._env.info_sets['landlord'].handcards_seq.copy(),
            'landlord_up': self._env.info_sets['landlord_up'].handcards_seq.copy(),
            'landlord_down': self._env.info_sets['landlord_down'].handcards_seq.copy(),
        }
        
        max_len = max(len(hand_seq['landlord']), len(hand_seq['landlord_up']), len(hand_seq['landlord_down']))
        
        # 填充到相同长度
        for pos in hand_seq:
            while len(hand_seq[pos]) < max_len:
                if len(hand_seq[pos]) > 0:
                    hand_seq[pos].append(hand_seq[pos][-1])
                else:
                    hand_seq[pos].append([])
        
        # 缓存函数
        def get_cached_min_steps(handcards):
            cache_key = (tuple(sorted(handcards)), tuple(sorted(self.mastercard_list)))
            if cache_key not in self._oracle_state['cache']:
                self._oracle_state['cache'][cache_key] = get_min_steps_to_win_with_mastercards(handcards, self.mastercard_list)
            return self._oracle_state['cache'][cache_key]
        
        # 只计算新的步骤
        start_idx = max(0, self._oracle_state['last_computed_step'] + 1)
        
        if start_idx < max_len:
            # 计算新步骤的oracle values
            new_N_landlord = [get_cached_min_steps(hand_seq['landlord'][i]) for i in range(start_idx, max_len)]
            new_N_peas1 = [get_cached_min_steps(hand_seq['landlord_up'][i]) for i in range(start_idx, max_len)]
            new_N_peas2 = [get_cached_min_steps(hand_seq['landlord_down'][i]) for i in range(start_idx, max_len)]
            
            # 计算新的advantages
            new_advantages = [nl - min(np1, np2) for nl, np1, np2 in zip(new_N_landlord, new_N_peas1, new_N_peas2)]
            
            # 更新缓存的advantages
            self._oracle_state['cached_advantages'].extend(new_advantages)
            self._oracle_state['last_computed_step'] = max_len - 1
        
        # 计算delta advantages
        advantages = self._oracle_state['cached_advantages']
        delta_adv = [advantages[0]] + [advantages[i] - advantages[i-1] for i in range(1, len(advantages))]
        
        # 分配rewards
        rewards = []
        step_positions = getattr(self._env, 'step_positions', [])
        order = ['landlord', 'landlord_up', 'landlord_down']
        
        if len(step_positions) == 0:
            for i, da in enumerate(delta_adv):
                pos = order[i % 3]
                coef = -1.0 if pos == 'landlord' else 0.5
                rewards.append(coef * da)
        else:
            for i, da in enumerate(delta_adv):
                if i < len(step_positions):
                    pos = step_positions[i]
                    coef = -1.0 if pos == 'landlord' else 0.5
                    rewards.append(coef * da)
                else:
                    rewards.append(0.0)
        
        return rewards


    # 优化版本3：近似快速计算

    def _get_oracle_reward_fast_approximation(self):
        """
        快速近似版本：使用简化的启发式代替expensive的oracle计算
        """
        hand_seq = {
            'landlord': self._env.info_sets['landlord'].handcards_seq.copy(),
            'landlord_up': self._env.info_sets['landlord_up'].handcards_seq.copy(),
            'landlord_down': self._env.info_sets['landlord_down'].handcards_seq.copy(),
        }
        
        max_len = max(len(hand_seq['landlord']), len(hand_seq['landlord_up']), len(hand_seq['landlord_down']))
        
        # 填充序列
        for pos in hand_seq:
            while len(hand_seq[pos]) < max_len:
                if len(hand_seq[pos]) > 0:
                    hand_seq[pos].append(hand_seq[pos][-1])
                else:
                    hand_seq[pos].append([])
        
        # 快速启发式：基于手牌数量和类型的简单估计
        def fast_estimate_min_steps(handcards):
            if len(handcards) == 0:
                return 0
            
            # 简单启发式：基于手牌数量，考虑mastercard的加成
            base_steps = len(handcards) // 3 + (1 if len(handcards) % 3 > 0 else 0)
            
            # mastercard 减少步数
            mastercard_count = sum(1 for card in handcards if card in self.mastercard_list)
            mastercard_bonus = min(mastercard_count, 2)  # 最多减少2步
            
            return max(1, base_steps - mastercard_bonus)
        
        # 快速计算
        N_landlord = [fast_estimate_min_steps(h) for h in hand_seq['landlord']]
        N_peas1 = [fast_estimate_min_steps(h) for h in hand_seq['landlord_up']]
        N_peas2 = [fast_estimate_min_steps(h) for h in hand_seq['landlord_down']]
        
        # 计算advantage
        Adv = [nl - min(np1, np2) for nl, np1, np2 in zip(N_landlord, N_peas1, N_peas2)]
        
        # 计算delta reward
        delta_adv = [Adv[0]] + [Adv[i] - Adv[i-1] for i in range(1, len(Adv))]
        
        # 分配rewards
        rewards = []
        step_positions = getattr(self._env, 'step_positions', [])
        order = ['landlord', 'landlord_up', 'landlord_down']
        
        if len(step_positions) == 0:
            for i, da in enumerate(delta_adv):
                pos = order[i % 3]
                coef = -1.0 if pos == 'landlord' else 0.5
                rewards.append(coef * da)
        else:
            for i, da in enumerate(delta_adv):
                if i < len(step_positions):
                    pos = step_positions[i]
                    coef = -1.0 if pos == 'landlord' else 0.5
                    rewards.append(coef * da)
                else:
                    rewards.append(0.0)
        
        return rewards


    # 优化版本4：混合策略 - 高频快速计算 + 低频精确计算

    def _get_oracle_reward_hybrid(self):
        """
        混合策略：大部分时间使用快速近似，偶尔使用精确计算来校正
        """
        # 每10个episode做一次精确计算，其他时候用快速近似
        if not hasattr(self, '_hybrid_counter'):
            self._hybrid_counter = 0
        
        self._hybrid_counter += 1
        
        if self._hybrid_counter % 10 == 0:
            # 每10次使用精确计算
            return self._get_oracle_reward_cached()
        else:
            # 其他时候使用快速近似
            return self._get_oracle_reward_fast_approximation()


    def _get_oracle_reward_fast(self):
        """
        快速版本的oracle reward计算
        只在必要时计算，使用缓存
        """
        if hasattr(self, '_cached_oracle_rewards'):
            return self._cached_oracle_rewards
        
        try:
            # 尝试快速计算
            hand_seq = {
                'landlord': self._env.info_sets['landlord'].handcards_seq[-10:],  # 只取最后10步
                'landlord_up': self._env.info_sets['landlord_up'].handcards_seq[-10:],
                'landlord_down': self._env.info_sets['landlord_down'].handcards_seq[-10:],
            }
            
            # 简化的reward计算
            rewards = []
            for i in range(len(hand_seq['landlord'])):
                landlord_cards = len(hand_seq['landlord'][i]) if i < len(hand_seq['landlord']) else 0
                peasant_cards = min(
                    len(hand_seq['landlord_up'][i]) if i < len(hand_seq['landlord_up']) else 20,
                    len(hand_seq['landlord_down'][i]) if i < len(hand_seq['landlord_down']) else 20
                )
                
                # 简单的advantage = 农民卡数 - 地主卡数
                advantage = peasant_cards - landlord_cards
                rewards.append(advantage * 0.1)  # 缩放因子
            
            self._cached_oracle_rewards = rewards
            return rewards
            
        except Exception as e:
            print(f"Fast oracle reward failed: {e}")
            # 返回sparse reward
            final_reward = self._get_reward()
            return [0.0] * (len(self._env.info_sets['landlord'].handcards_seq) - 1) + [final_reward]

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action

def get_obs(infoset,resnet_model=False):
    """
    This function obtains observations with imperfect information
    from the infoset. It has three branches since we encode
    different features for different positions.
    
    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be landlord/landlord_down/landlord_up

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """
    if resnet_model==True:
        return _get_obs_resnet(infoset)
    else:
        if infoset.player_position == 'landlord':
            return _get_obs_landlord(infoset)
        elif infoset.player_position == 'landlord_up':
            return _get_obs_landlord_up(infoset)
        elif infoset.player_position == 'landlord_down':
            return _get_obs_landlord_down(infoset)
        else:
            raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(ACTION_ENCODE_DIM, dtype=np.int8)

    matrix = np.zeros([ENCODE_DIM, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)

    for card, num_times in counter.items():
            
        if card < 20:
            matrix[0:4, Card2Column[card]] = NumOnes2Array[num_times]
            # if is_mastercard(card):  
            #     matrix[4, Card2Column[card]] = 1
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    # print("global_mastercard_values: ", global_mastercard_values)
    for mcard in global_mastercard_values:
        matrix[4, Card2Column[mcard]] = 1
    # print("matrix: ", global_mastercard_values,matrix.shape, matrix)
    return np.concatenate((matrix.flatten('F'), jokers))


def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5xACTION_ENCODE_DIM matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), ACTION_ENCODE_DIM))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 3*ACTION_ENCODE_DIM)
    return action_seq_array


def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """

    player_id = 'landlord'
    prev_id = 'landlord_up'
    next_id = 'landlord_down'

    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)
    # print("my_action_batch : ",my_action_batch.shape, my_action_batch)


    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    
    # NEW: 
    control_flag = 1.0 if infoset.last_pid == player_id else 0.0
    control_flag = np.array(control_flag).reshape(1)
    control_flag_batch = np.repeat(control_flag[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    prev_hand = _cards2array(infoset.all_handcards[prev_id])
    prev_hand_batch = np.repeat(prev_hand[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    next_hand = _cards2array(infoset.all_handcards[next_id])
    next_hand_batch = np.repeat(next_hand[np.newaxis, :],
                                   num_legal_actions, axis=0) 

    x_batch = np.hstack((my_handcards_batch, # 67
                         other_handcards_batch,# 67
                         last_action_batch,# 67
                         landlord_up_played_cards_batch,# 67
                         landlord_down_played_cards_batch,# 67
                         landlord_up_num_cards_left_batch,# 17
                         landlord_down_num_cards_left_batch,# 17
                         bomb_num_batch,# 15
                         control_flag_batch, # 1 
                         my_action_batch)) # 67
    
  
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,    
                             bomb_num,
                             control_flag,))
    
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    x_addition_batch = np.hstack([
        prev_hand_batch,
        next_hand_batch,
    ])
    x_addition = np.hstack([
        prev_hand,
        next_hand,
    ])


    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
            'x_addition': x_addition.astype(np.int8), # <= NEW
            'x_addition_batch': x_addition_batch.astype(np.float32), # <= NEW
          }
    return obs
def _get_obs_landlord_up(infoset):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    player_id = 'landlord_up'
    prev_id = 'landlord_down'
    next_id = 'landlord'

    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_down'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    
    # NEW: 
    control_flag = 1.0 if infoset.last_pid == player_id else 0.0
    control_flag = np.array(control_flag).reshape(1)
    control_flag_batch = np.repeat(control_flag[np.newaxis, :],
                                   num_legal_actions, axis=0)

    prev_hand = _cards2array(infoset.all_handcards[prev_id])
    prev_hand_batch = np.repeat(prev_hand[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    next_hand = _cards2array(infoset.all_handcards[next_id])
    next_hand_batch = np.repeat(next_hand[np.newaxis, :],
                                   num_legal_actions, axis=0) 

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         control_flag_batch,
                         my_action_batch))
   
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num,
                             control_flag))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))

    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    x_addition_batch = np.hstack([
        prev_hand_batch,
        next_hand_batch,
    ])
    x_addition = np.hstack([
        prev_hand,
        next_hand,
    ])


    obs = {
            'position': 'landlord_up',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
            'x_addition': x_addition.astype(np.int8), # <= NEW
            'x_addition_batch': x_addition_batch.astype(np.float32), # <= NEW
          }
    return obs

def _get_obs_landlord_down(infoset):
    """
    Obttain the landlord_down features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    player_id = 'landlord_down'
    prev_id = 'landlord'
    next_id = 'landlord_up'
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)
    
    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_up'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    
    # NEW: 
    control_flag = 1.0 if infoset.last_pid == player_id else 0.0
    control_flag = np.array(control_flag).reshape(1)
    control_flag_batch = np.repeat(control_flag[np.newaxis, :],
                                   num_legal_actions, axis=0)

    prev_hand = _cards2array(infoset.all_handcards[prev_id])
    prev_hand_batch = np.repeat(prev_hand[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    next_hand = _cards2array(infoset.all_handcards[next_id])
    next_hand_batch = np.repeat(next_hand[np.newaxis, :],
                                   num_legal_actions, axis=0)                               

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         control_flag_batch,
                         my_action_batch))

   
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num,
                             control_flag))
    
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    
    x_addition_batch = np.hstack([
        prev_hand_batch,
        next_hand_batch,
    ])
    x_addition = np.hstack([
        prev_hand,
        next_hand,
    ])

    # print("down x_addition_batch.shape: ", x_addition_batch.shape)
    # print("down x_batch.shape: ", x_batch.shape)

    obs = {
            'position': 'landlord_down',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
            'x_addition': x_addition.astype(np.int8), # <= NEW
            'x_addition_batch': x_addition_batch.astype(np.float32), # <= NEW
          }
    return obs


def set_global_mastercards(mastercard_list):
    """Set the global mastercard values, ensuring jokers are not included"""
    global global_mastercard_values
    # Filter out jokers (20 and 30) from mastercard list
    global_mastercard_values = [card for card in mastercard_list if card < 20]
    

def is_mastercard(card):
    """Check if a card is a mastercard (jokers are never mastercards)."""
    return card in global_mastercard_values and card < 20  # Jokers (20,30) are never mastercards


def _get_obs_resnet(infoset):
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)

    three_landlord_cards = _cards2array(infoset.three_landlord_cards)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

 

    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    
    num_cards_left = np.hstack((
                         landlord_num_cards_left,  # 20
                         landlord_up_num_cards_left,  # 17
                         landlord_down_num_cards_left))

    x_batch = np.hstack((
                        #  bid_info_batch,  # 3
                         bomb_num_batch,  # 15
                         ))
    x_no_action = np.hstack((
                            #  bid_info,
                             bomb_num,
                             ))

    z = np.vstack((
                  num_cards_left,  # 54
                  my_handcards,  # 54
                  other_handcards,  # 54
                  three_landlord_cards,  # 54
                  landlord_played_cards,  # 54
                  landlord_up_played_cards,  # 54
                  landlord_down_played_cards,  # 54
                  _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq, 60))
                  ))


    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:, np.newaxis, :]
    z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)

    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs