"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(201, 128, batch_first=True)
        self.dense1 = nn.Linear(384+67 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        # print("shapes", z.shape, lstm_out.shape, x.shape)
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
        self.lstm = nn.LSTM(201, 128, batch_first=True)
        self.dense1 = nn.Linear(521 + 67 + 128, 512)
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

# # Model dict is only used in evaluation but not training
# model_dict = {}
# model_dict['landlord'] = LandlordLstmModel
# model_dict['landlord_up'] = FarmerLstmModel
# model_dict['landlord_down'] = FarmerLstmModel

# class Model:
#     """
#     The wrapper for the three models. We also wrap several
#     interfaces such as share_memory, eval, etc.
#     """
#     def __init__(self, device=0):
#         self.models = {}
#         if not device == "cpu":
#             device = 'cuda:' + str(device)
#         self.models['landlord'] = LandlordLstmModel().to(torch.device(device))
#         self.models['landlord_up'] = FarmerLstmModel().to(torch.device(device))
#         self.models['landlord_down'] = FarmerLstmModel().to(torch.device(device))

#     def forward(self, position, z, x, training=False, flags=None):
#         model = self.models[position]
#         return model.forward(z, x, training, flags)

#     def share_memory(self):
#         self.models['landlord'].share_memory()
#         self.models['landlord_up'].share_memory()
#         self.models['landlord_down'].share_memory()

#     def eval(self):
#         self.models['landlord'].eval()
#         self.models['landlord_up'].eval()
#         self.models['landlord_down'].eval()

#     def parameters(self, position):
#         return self.models[position].parameters()

#     def get_model(self, position):
#         return self.models[position]

#     def get_models(self):
#         return self.models


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlockM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockM, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.mish = nn.Mish(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = ChannelAttention(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.mish(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += self.shortcut(x)
        out = self.mish(out)
        return out


class GeneralModelResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 68
        self.layer1 = self._make_layer(BasicBlockM, 68, 3, stride=2)  # 1*34*68,34=67//2
        self.layer2 = self._make_layer(BasicBlockM, 68*2, 3, stride=2)  # 1*17*68*2
        self.layer3 = self._make_layer(BasicBlockM, 68*4, 3, stride=2)  # 1*9*68*4
        self.linear1 = nn.Linear(68*4 * BasicBlockM.expansion * 9 + 15 * 4, 2048) #x_batch dim is 15
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            # print("self.in_planes", self.in_planes)
        return nn.Sequential(*layers)

    # def check_no_bombs(self, a):
    #     reshaped_a = a[:52].view(-1, 4)
    #     sums = reshaped_a.sum(dim=1)
    #     if torch.all(sums < 4) and a[52] + a[53] < 2:
    #         return True
    #     else:
    #         return False
    
    def check_no_bombs(self, a):
        sub_a = a[:13*5]  # Take first 65 elements
        reshaped_a = sub_a.reshape(5, -1)

        # Get the 5th row (mastercard label)
        fifth_row = reshaped_a[4,:]

        # Find columns where 5th row is 0 and 1
        zero_cols = (fifth_row == 0)
        one_cols = (fifth_row == 1)

        zero_cols_sum = reshaped_a[:, zero_cols].sum()
        one_cols_sum = reshaped_a[:, one_cols].sum()
        total_sum = zero_cols_sum + one_cols_sum

        # cond1 = (total_sum >= 4)       # tensor bool, shape: scalar
        # cond2 = (a[-1] + a[-2] == 2)   # tensor bool, shape: scalar

        # cond = cond1 | cond2           # 逻辑或（仍然是 tensor bool）
        # return cond

        if torch.all(total_sum < 4) and (a[-1] + a[-2] < 2):
            return True
        else:
            return False
      
    
    def forward(self, z, x, return_value=False, flags=None, debug=False):

        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1, 2)
        out = torch.cat([x, x, x, x, out], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = self.linear4(out)
        win_rate, win, lose = torch.split(out, (1, 1, 1), dim=-1)
        win_rate = torch.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        bombs = True
        if self.check_no_bombs(z[0, 2]) and self.check_no_bombs(z[0, 3]) and (0 in z[0, 11]):
            bombs = False
            out = _win_rate
        else:
            out = _win_rate * win + (1. - _win_rate) * lose
        
        # print(f"win_rate type: {type(win_rate)}, win type: {type(win)}, lose type: {type(lose)}")

        if return_value:
            return dict(values=(win_rate, win, lose))
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            elif flags is not None and flags.action_threshold > 0 and bombs:
                max_adp = torch.max(out)
                if max_adp >= 0:
                    min_threshold = max_adp * (1 - flags.action_threshold)
                else:
                    min_threshold = max_adp * (1 + flags.action_threshold)
                valid_indices = torch.where(out >= min_threshold)[0]
                action = valid_indices[torch.argmax(_win_rate[valid_indices])]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out), values=out)


model_dict = {
    "landlord": GeneralModelResnet,
    "landlord_down": GeneralModelResnet,
    "landlord_up": GeneralModelResnet,
}



class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        if not device == "cpu":
            device = 'cuda:' + str(device)

        self.models = {
            'landlord': GeneralModelResnet().to(torch.device(device)),
            'landlord_down': GeneralModelResnet().to(torch.device(device)),
            'landlord_up': GeneralModelResnet().to(torch.device(device)),
        }

    def forward(self, position, z, x, training=False, flags=None, debug=False):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['landlord_up'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_down'].eval()
        self.models['landlord_up'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
