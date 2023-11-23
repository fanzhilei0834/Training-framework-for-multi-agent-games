import torch
import torch.nn.functional as F
import pickle
import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from .ResnetModel import Resnet18

base_dir = os.path.dirname(os.path.abspath(__file__))
# print(base_dir)
net = Resnet18()
net = torch.nn.DataParallel(net)
# net.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + '/weight.pkl', map_location=torch.device('cpu')))
with open(os.path.dirname(os.path.abspath(__file__)) + '/weight0.pkl', "rb") as fwei:
    net_weight = pickle.load(fwei)
    net.load_state_dict(net_weight)
    net.eval()

sys.path.pop(-1)  # just for safety

def my_controller(observation, action_space=None, is_act_continuous=False):
    # print('\n')
    # print("controlled_player_index: ", observation['controlled_player_index'], "current_move_player: ", observation['current_move_player'])
    # print("observation in SL: ", observation['obs']['observation'].shape if observation['obs'] else None)
    # print("action_mask in SL: ", observation['obs']['action_mask'] if observation['obs'] else None)
    action = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    if not observation['obs']:
        action[0][-1] = 1
        # print("agent_action: ", action)
        # print('\n')
        return action
    
    obs = convert(observation['obs']['observation'], observation['controlled_player_index']).unsqueeze(0)
    legactions = torch.tensor(observation['obs']['action_mask'], dtype=torch.float).unsqueeze(0)
    res, value = net(obs, legactions)
    pi = F.softmax(res, dim=-1)
    act = torch.argmax(pi, dim=-1, keepdim=True).item()
    action[0][act] = 1

    # print("agent_action: ", action)
    # print('\n')
    return action

def convert(observation, player_index):
    hand_card = torch.zeros(4, 4, 9)  # (channel, row, col)
    for i in range(34):
        for j in range(4):
            if observation[0][i][j] == 1:
                x = i // 9
                y = i % 9
                hand_card[j][x][y] = 1

    remain = torch.zeros(4, 4, 9)
    for i in range(34):
        for j in range(4):
            if observation[1][i][j] == 1:
                x = i // 9
                y = i % 9
                remain[j][x][y] = 1
    
    show_self = torch.zeros(4, 4, 9)
    for i in range(34):
        for j in range(4):
            if observation[player_index+2][i][j] == 1:
                x = i // 9
                y = i % 9
                show_self[j][x][y] = 1
    
    show_xj = torch.zeros(4, 4, 9)
    for i in range(34):
        for j in range(4):
            if observation[(player_index + 1) % 4 + 2][i][j] == 1:
                x = i // 9
                y = i % 9
                show_xj[j][x][y] = 1
    
    show_dj = torch.zeros(4, 4, 9)
    for i in range(34):
        for j in range(4):
            if observation[(player_index + 2) % 4 + 2][i][j] == 1:
                x = i // 9
                y = i % 9
                show_dj[j][x][y] = 1
    
    show_sj = torch.zeros(4, 4, 9)
    for i in range(34):
        for j in range(4):
            if observation[(player_index + 3) % 4 + 2][i][j] == 1:
                x = i // 9
                y = i % 9
                show_sj[j][x][y] = 1
    
    return torch.cat((hand_card, remain, show_self, show_xj, show_dj, show_sj), 0)