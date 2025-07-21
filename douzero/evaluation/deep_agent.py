import torch
import numpy as np
from collections import Counter
# from douzero.env.env import get_obs
import douzero.env.env as env
import douzero.env.env_alpha as env_alpha

def _load_model(position, model_path):
    # if 'weights' in model_path:
    #     from douzero.dmc.models import model_dict
    #     model = model_dict[position]()
    #     model_state_dict = model.state_dict()
    # else:
    #     from douzero.dmc.model_alpha import model_dict
    #     model = model_dict[position]()
    #     model_state_dict = model.state_dict()
    from douzero.dmc.models import model_dict
    model = model_dict[position]()
    model_state_dict = model.state_dict()

    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


class DeepAgent:

    def __init__(self, position, model_path,mode='eval'):
        self.model = _load_model(position, model_path)

        self.mode = mode

    def act(self, infoset):

        # if len(infoset.legal_actions) == 1:
        #     return infoset.legal_actions[0]

        obs = env.get_obs(infoset) 

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        x_addition_batch = torch.from_numpy(obs['x_addition_batch']).float()
        

        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        if torch.cuda.is_available():
            z_batch, x_batch,x_addition_batch = z_batch.cuda(), x_batch.cuda(), x_addition_batch.cuda(),
        # z_batch, x_batch,x_addition_batch = z_batch.to(self.device), x_batch.to(self.device), x_addition_batch.to(self.device),
        
        y_pred = self.model.forward(z_batch, x_batch, x_addition_batch,return_value=True,mode=self.mode)['actor_value']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action


class DeepAgent_Alpha:

    def __init__(self, position, model_path):
        if "test" in model_path:
            self.model_type = "test"
        elif "best" in model_path:
            self.model_type = "best"
        else:
            self.model_type = "new"
        self.model = _load_model(position, model_path)
        
        
    def check_no_bombs_old(self, cards):
        card_counts = Counter(cards)
        for count in card_counts.values():
            if count == 4:
                return False

        if 20 in card_counts and 30 in card_counts:
            return False

        return True
    
    def check_no_bombs(self, cards,mastercards=None):
        card_counts = Counter(cards)
        mastercard_counts = 0
        for card in cards:
            if card in mastercards:
                mastercard_counts += 1

        for count in card_counts.values():
            if count + mastercard_counts >= 4:
                return False
        if 20 in card_counts and 30 in card_counts:
            return False
        return True

    def act(self, infoset):
        # if self.model_type == "test":
        #     obs = get_obs_douzero(infoset)
        # elif self.model_type == "best":
        #     obs = _get_obs_resnet(infoset, infoset.player_position)
        # else:
  
        obs = env_alpha.get_obs(infoset, new_model=True)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()

        if self.model_type != 'new':
            y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        else:
            win_rate, win, lose = self.model.forward(z_batch, x_batch, return_value=True)['values']
            if infoset.player_position in ["landlord", "landlord_up", "landlord_down"]:
                _win_rate = (win_rate + 1) / 2
                y_pred = _win_rate * win + (1. - _win_rate) * lose
                _win_rate = _win_rate.detach().cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()        

                if self.check_no_bombs(infoset.player_hand_cards,infoset.mastercard_list) and self.check_no_bombs(
                        infoset.other_hand_cards,infoset.mastercard_list):
                    best_action_index = np.argmax(_win_rate, axis=0)[0]
                    best_action = infoset.legal_actions[best_action_index]
                else:
                    y_pred = y_pred.flatten()
                    _win_rate = _win_rate.flatten()
                    max_adp = np.max(y_pred)
                    if max_adp >= 0:
                        min_threshold = max_adp * 0.95
                    else:
                        min_threshold = max_adp * 1.05
                    valid_indices = np.where(y_pred >= min_threshold)[0]
                    best_action_index = valid_indices[np.argmax(_win_rate[valid_indices])]
                    best_action = infoset.legal_actions[best_action_index]
                return best_action
            else:
                y_pred = win_rate[:, :1] * win + win_rate[:, 1:2] * lose
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        return best_action


