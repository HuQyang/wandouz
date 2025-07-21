"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn

from douzero.env.utils import ACTION_ENCODE_DIM, ENCODE_DIM


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(ACTION_ENCODE_DIM*3, 128, batch_first=True)
        self.dense1 = nn.Linear(38452+ACTION_ENCODE_DIM + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(ACTION_ENCODE_DIM*3, 128, batch_first=True)
        self.dense1 = nn.Linear(589 + ACTION_ENCODE_DIM + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]

            return dict(action=action)



import torch.nn.functional as F
from torch.distributions import Categorical

class LandlordPerfectDou(nn.Module):
    def __init__(self,
                 input_size=ACTION_ENCODE_DIM*3,
                 lstm_hidden_size=128,
                 lstm_num_layers=1,
                 imperct_dim = 452,
                 action_dim=1,
                 critic_feature_dim = 452+ACTION_ENCODE_DIM*2,
                 actor_mlp_sizes=[512,512,512,512,512,512, 1],
                 critic_mlp_sizes=[512,512,512,512,512,512, 1]):
        super().__init__()
        
        
        self.actor_lstm = nn.LSTM(input_size,
                                   lstm_hidden_size,
                                   num_layers=lstm_num_layers,
                                   batch_first=True)
        
        self.critic_lstm = nn.LSTM(input_size,
                                   lstm_hidden_size,
                                   num_layers=lstm_num_layers,
                                   batch_first=True)
        # self.shared_lstm = nn.LSTM(input_size,
        #                            lstm_hidden_size,
        #                            num_layers=lstm_num_layers,
        #                            batch_first=True)
        

        # Actor 分支
        # actor_layers = []
        # in_dim = lstm_hidden_size + imperct_dim  
        # for h in actor_mlp_sizes:
        #     actor_layers += [nn.Linear(in_dim, h), nn.ReLU()]
        #     in_dim = h
        # actor_layers.append(nn.Linear(in_dim, action_dim))
        # self.actor_head = nn.Sequential(*actor_layers)

        # # Critic 分支：接收 LSTM 输出 + 完美信息特征
        # critic_layers = []
        # in_dim = lstm_hidden_size + critic_feature_dim
        # for h in critic_mlp_sizes:
        #     critic_layers += [nn.Linear(in_dim, h), nn.ReLU()]
        #     in_dim = h
        # critic_layers.append(nn.Linear(in_dim, 1))
        # self.critic_head = nn.Sequential(*critic_layers)

        actor_in_dim = lstm_hidden_size + imperct_dim 
        self.actor_lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        self.dense1 = nn.Linear(actor_in_dim, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)
        

        critic_in_dim = lstm_hidden_size + critic_feature_dim
        # print("critic_in_dim", critic_in_dim)
        self.critic_lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        self.critic_dense1 = nn.Linear(critic_in_dim, 512)
        self.critic_dense2 = nn.Linear(512, 512)
        self.critic_dense3 = nn.Linear(512, 512)
        self.critic_dense4 = nn.Linear(512, 512)
        self.critic_dense5 = nn.Linear(512, 512)
        self.critic_dense6 = nn.Linear(512, 1)

    def forward(self, z, x, x_addition, return_value=False,flags=None,mode='train'):
        # """
        # incomplete_feats: [B, T, input_size]  不完全信息序列
        # perfect_feats:    [B, critic_feature_dim] 完美信息向量
        # """

        # lstm_out, _ = self.actor_lstm(z)
        # lstm_out = lstm_out[:, -1, :]   # 取最后时刻输出
        # emb_act = torch.cat([lstm_out,x], dim=-1)
        # actor_value = self.actor_head(emb_act)              # [B, action_dim]
        # # probs  = F.softmax(logits, dim=-1)         # categorical probs

        # lstm_out, _ = self.critic_lstm(z)
        # lstm_out = lstm_out[:, -1, :]   # 取最后时刻输出

        # x_perfect = torch.cat([x, x_addition], dim=-1)
        # critic_in = torch.cat([lstm_out, x_perfect], dim=-1)
        # critic_value     = self.critic_head(critic_in).squeeze(-1)  # [B]
        if mode == 'train':
            lstm_out, _ = self.actor_lstm(z)
        else:
            lstm_out, _ = self.critic_lstm(z)
        lstm_out = lstm_out[:, -1, :]   # 取最后时刻输出
        
        emb_act = torch.cat([lstm_out,x], dim=-1)
        
        emb_act = self.dense1(emb_act)
        emb_act = torch.relu(emb_act)
        emb_act = self.dense2(emb_act)
        emb_act = torch.relu(emb_act)
        emb_act = self.dense3(emb_act)
        emb_act = torch.relu(emb_act)
        emb_act = self.dense4(emb_act)
        emb_act = torch.relu(emb_act)
        emb_act = self.dense5(emb_act)
        emb_act = torch.relu(emb_act)
        actor_value = self.dense6(emb_act)
        # actor_value = torch.sigmoid(actor_value)  # Ensure the output is in [0, 1] range
        
        cri_lstm_out, _ = self.critic_lstm(z)
        cri_lstm_out = cri_lstm_out[:, -1, :]   # 取最后时刻输出

        x_perfect = torch.cat([x, x_addition], dim=-1)

        emb_cri = torch.cat([cri_lstm_out,x_perfect], dim=-1)
        
        emb_cri = self.critic_dense1(emb_cri)
        emb_cri = torch.relu(emb_cri)
        emb_cri = self.critic_dense2(emb_cri)
        emb_cri = torch.relu(emb_cri)
        emb_cri = self.critic_dense3(emb_cri)
        emb_cri = torch.relu(emb_cri)
        emb_cri = self.critic_dense4(emb_cri)
        emb_cri = torch.relu(emb_cri)
        emb_cri = self.critic_dense5(emb_cri)
        emb_cri = torch.relu(emb_cri)
        critic_value = self.critic_dense6(emb_cri)

        if return_value:
            return dict(critic_value=critic_value,actor_value=actor_value)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(actor_value.shape[0], (1,))[0]
            else:
                action = torch.argmax(actor_value,dim=0)[0]

            return dict(action=action,critic_value=critic_value,actor_value=actor_value)



class FarmerPerfectDou(nn.Module):
    def __init__(self,
                 input_size=ACTION_ENCODE_DIM*3,
                 lstm_hidden_size=128,
                 lstm_num_layers=1,
                 imperct_dim = 589,
                 action_dim=1,
                 critic_feature_dim = 589+ACTION_ENCODE_DIM*2,
                 actor_mlp_sizes=[512,512,512,512,512,512, 1],
                 critic_mlp_sizes=[512,512,512,512,512,512, 1]):
        super().__init__()
        self.actor_lstm = nn.LSTM(input_size,
                                   lstm_hidden_size,
                                   num_layers=lstm_num_layers,
                                   batch_first=True)
        
        self.critic_lstm = nn.LSTM(input_size,
                                   lstm_hidden_size,
                                   num_layers=lstm_num_layers,
                                   batch_first=True)
        # self.shared_lstm = nn.LSTM(input_size,
        #                            lstm_hidden_size,
        #                            num_layers=lstm_num_layers,
        #                            batch_first=True)
        

        # Actor 分支
        actor_layers = []
        actor_in_dim = lstm_hidden_size + imperct_dim
        # for h in actor_mlp_sizes:
        #     actor_layers += [nn.Linear(actor_in_dim, h), nn.ReLU()]
        #     actor_in_dim = h
        # actor_layers.append(nn.Linear(actor_in_dim, action_dim))
        # self.actor_head = nn.Sequential(*actor_layers)

        self.actor_lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        self.dense1 = nn.Linear(actor_in_dim, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

        critic_in_dim = lstm_hidden_size + critic_feature_dim
        # print("critic_in_dim", critic_in_dim)
        self.critic_lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        self.critic_dense1 = nn.Linear(critic_in_dim, 512)
        self.critic_dense2 = nn.Linear(512, 512)
        self.critic_dense3 = nn.Linear(512, 512)
        self.critic_dense4 = nn.Linear(512, 512)
        self.critic_dense5 = nn.Linear(512, 512)
        self.critic_dense6 = nn.Linear(512, 1)

        # Critic 分支：接收 LSTM 输出 + 完美信息特征
        # critic_layers = []
        # in_dim = lstm_hidden_size + critic_feature_dim
        # for h in critic_mlp_sizes:
        #     critic_layers += [nn.Linear(in_dim, h), nn.ReLU()]
        #     in_dim = h
        # critic_layers.append(nn.Linear(in_dim, 1))
        # self.critic_head = nn.Sequential(*critic_layers)
        

    def forward(self, z, x, x_addition, return_value=False, flags=None,mode='train'):
        if mode == 'train':
            lstm_out, _ = self.actor_lstm(z)
        else:
            lstm_out, _ = self.critic_lstm(z)
        lstm_out = lstm_out[:, -1, :]   # 取最后时刻输出
        
        emb_act = torch.cat([lstm_out,x], dim=-1)
        
        emb_act = self.dense1(emb_act)
        emb_act = torch.relu(emb_act)
        emb_act = self.dense2(emb_act)
        emb_act = torch.relu(emb_act)
        emb_act = self.dense3(emb_act)
        emb_act = torch.relu(emb_act)
        emb_act = self.dense4(emb_act)
        emb_act = torch.relu(emb_act)
        emb_act = self.dense5(emb_act)
        emb_act = torch.relu(emb_act)
        actor_value = self.dense6(emb_act)
        
        cri_lstm_out, _ = self.critic_lstm(z)
        cri_lstm_out = cri_lstm_out[:, -1, :]   # 取最后时刻输出

        x_perfect = torch.cat([x, x_addition], dim=-1)

        emb_cri = torch.cat([cri_lstm_out,x_perfect], dim=-1)
        
        emb_cri = self.critic_dense1(emb_cri)
        emb_cri = torch.relu(emb_cri)
        emb_cri = self.critic_dense2(emb_cri)
        emb_cri = torch.relu(emb_cri)
        emb_cri = self.critic_dense3(emb_cri)
        emb_cri = torch.relu(emb_cri)
        emb_cri = self.critic_dense4(emb_cri)
        emb_cri = torch.relu(emb_cri)
        emb_cri = self.critic_dense5(emb_cri)
        emb_cri = torch.relu(emb_cri)
        critic_value = self.critic_dense6(emb_cri)

        if return_value:
            return dict(critic_value=critic_value,actor_value=actor_value)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(actor_value.shape[0], (1,))[0]
            else:
                action = torch.argmax(actor_value,dim=0)[0]
           
            return dict(action=action,critic_value=critic_value,actor_value=actor_value)

def compute_gae_advantage(rewards, values, dones, gamma=1.0, lam=0.95):
    """
    Input:
      rewards: Tensor of shape [T], 每步即时奖励 r_t
      values:  Tensor of shape [T+1], 包括从 s_0 到 s_T 的 V_phi(s)
      dones:   Tensor of shape [T], 1 表示该步后游戏终止
      
    Returns:
      advantages: Tensor [T], 对应每步 A_t
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]   # 如果 done，则不把下一步的价值带入
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    return advantages

def select_action(agent, inc_feats, perf_feats, legal_actions, evaluate=False):
    with torch.no_grad():
        probs, value, logits = agent(inc_feats, perf_feats)
        dist = Categorical(logits=logits[:, legal_actions])
        if evaluate:
            rel_act = torch.argmax(dist.probs, dim=-1)
        else:
            rel_act = dist.sample()
        action = legal_actions[rel_act]
        logp   = dist.log_prob(rel_act)
    return action, logp, value

# Model dict is only used in evaluation but not training
# model_dict = {}
# model_dict['landlord'] = LandlordLstmModel
# model_dict['landlord_up'] = FarmerLstmModel
# model_dict['landlord_down'] = FarmerLstmModel

# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['landlord'] = LandlordPerfectDou
model_dict['landlord_up'] = FarmerPerfectDou
model_dict['landlord_down'] = FarmerPerfectDou


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord'] = LandlordPerfectDou().to(torch.device(device))
        self.models['landlord_up'] = FarmerPerfectDou().to(torch.device(device))
        self.models['landlord_down'] = FarmerPerfectDou().to(torch.device(device))

    def forward(self, position, z, x,x_addition, return_value=False, flags=None):
        model = self.models[position]
        return model.forward(z, x,x_addition, return_value, flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
