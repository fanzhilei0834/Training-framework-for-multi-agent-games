import torch
import torch.nn.functional as F
import os
import pickle
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from .model import MLPnet

base_dir = os.path.dirname(os.path.abspath(__file__))
# print(base_dir)
net = MLPnet()
with open(os.path.dirname(os.path.abspath(__file__)) + '/weight.pkl', "rb") as fwei:
    net_weight = pickle.load(fwei)
    net.load_state_dict(net_weight)
    net.eval()

sys.path.pop(-1)  # just for safety

def my_controller(observation, action_space=None, is_act_continuous=False):
    # print('\n')
    # print("controlled_player_index: ", observation['controlled_player_index'], "current_move_player: ", observation['current_move_player'])
    # print("observation in SL: ", observation['obs']['observation'].shape if observation['obs'] else None)
    # print("action_mask in SL: ", observation['obs']['action_mask'] if observation['obs'] else None)
    action = [[0] * 91]

    if not observation['obs']:
        action[0][-1] = 1
        # print("agent_action: ", action)
        # print('\n')
        return action
    
    obs = convert(observation['obs']['observation']).unsqueeze(0)
    legactions = torch.tensor(observation['obs']['action_mask'], dtype=torch.float).unsqueeze(0)
    res, value = net(obs, legactions)
    pi = F.softmax(res, dim=-1)
    act = torch.argmax(pi, dim=-1, keepdim=True).item()
    action[0][act] = 1

    # print("agent_action: ", action)
    # print('\n')
    return action

def convert(observation):
    return torch.tensor(observation, dtype=torch.float)
