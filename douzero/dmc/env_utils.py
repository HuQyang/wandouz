"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch 

# def _format_observation(obs, device):
#     """
#     A utility function to process observations and
#     move them to CUDA.
#     """
#     position = obs['position']
#     if not device == "cpu":
#         device = 'cuda:' + str(device)
#     device = torch.device(device)
#     x_batch = torch.from_numpy(obs['x_batch']).to(device)
#     z_batch = torch.from_numpy(obs['z_batch']).to(device)
#     x_no_action = torch.from_numpy(obs['x_no_action'])
#     z = torch.from_numpy(obs['z'])
#     x_addition = torch.from_numpy(obs['x_addition'])

#     x_addition_batch = torch.from_numpy(obs['x_addition_batch']).to(device)
#     advantage = torch.from_numpy(obs['advantage']).to(device)

#     obs = {'x_batch': x_batch,
#            'z_batch': z_batch,
#            'legal_actions': obs['legal_actions'],
#            'x_addition_batch': x_addition_batch,
#            'advantage': advantage,
#            }
#     return position, obs, x_no_action, z, x_addition

# class Environment:
#     def __init__(self, env, device):
#         """ Initialzie this environment wrapper
#         """
#         self.env = env
#         self.device = device
#         self.episode_return = None

#     def initial(self):
#         initial_position, initial_obs, x_no_action, z, x_addition = _format_observation(self.env.reset(), self.device)
#         initial_reward = torch.zeros(1, 1)
#         self.episode_return = torch.zeros(1, 1)
#         initial_done = torch.ones(1, 1, dtype=torch.bool)

#         return initial_position, initial_obs, dict(
#             done=initial_done,
#             episode_return=self.episode_return,
#             obs_x_no_action=x_no_action,
#             obs_z=z,
#             obs_x_addition=x_addition,
#             )
        
#     def step(self, action):
#         obs, reward, done, _ = self.env.step(action)

#         self.episode_return += reward
#         episode_return = self.episode_return 

#         if done:
#             obs = self.env.reset()
#             self.episode_return = torch.zeros(1, 1)

#         position, obs, x_no_action, z, x_addition = _format_observation(obs, self.device)
#         reward = torch.tensor(reward).view(1, 1)
#         done = torch.tensor(done).view(1, 1)
        
#         return position, obs, dict(
#             done=done,
#             episode_return=episode_return,
#             obs_x_no_action=x_no_action,
#             obs_z=z,
#             obs_x_addition=x_addition,
#             )

#     def close(self):
#         self.env.close()

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    if obs is None:
        # 如果obs是None（游戏结束），返回None
        return None, None, None, None, None
        
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    x_addition = torch.from_numpy(obs['x_addition'])

    x_addition_batch = torch.from_numpy(obs['x_addition_batch']).to(device)
    # 初始化advantage为0，后续会在act函数中计算
    advantage = torch.zeros(1)

    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           'x_addition_batch': x_addition_batch,
           'advantage': advantage,
           }
    return position, obs, x_no_action, z, x_addition

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        initial_position, initial_obs, x_no_action, z, x_addition = _format_observation(self.env.reset(), self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            obs_x_addition=x_addition,
            )
        
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        # 修复关键问题：处理reward的类型
        if isinstance(reward, list):
            # 如果reward是列表，取最后一个值
            if len(reward) > 0:
                current_reward = reward[-1]
            else:
                current_reward = 0.0
        elif isinstance(reward, np.ndarray):
            # 如果reward是numpy数组
            current_reward = float(reward.item()) if reward.size == 1 else float(reward[-1])
        else:
            # 如果reward已经是数值
            current_reward = float(reward)
        
        # 确保reward是tensor格式
        reward_tensor = torch.tensor(current_reward, dtype=torch.float32).view(1, 1)
        
        self.episode_return += reward_tensor
        episode_return = self.episode_return 

        if done:
            # 游戏结束时重置环境
            self.episode_return = torch.zeros(1, 1)
            obs = self.env.reset()

        # 关键修复：确保obs不为None
        if obs is None:
            # 如果obs是None，强制重置环境
            obs = self.env.reset()
            
        position, obs, x_no_action, z, x_addition = _format_observation(obs, self.device)
        reward = reward_tensor  # 返回tensor格式的reward
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            obs_x_addition=x_addition,
            )

    def close(self):
        self.env.close()