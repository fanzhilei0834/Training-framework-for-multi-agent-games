# AAMAS2023 竞赛四人麻将赛道 RL 自博弈 localgame 部分
# author: fanzl
# date: 2023/4/14
# version: 1.1
# 版本改进：
# 待完善:


import time
import random
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

from env.chooseenv import make
from ResnetModel import Resnet18, load, convert2


def getAction(observation):
    global net
    action = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    if not observation['obs']:
        action[0][-1] = 1
        return action
    
    obs = convert2(observation['obs']['observation'], observation['controlled_player_index']).unsqueeze(0)
    legactions = torch.tensor(observation['obs']['action_mask'], dtype=torch.float).unsqueeze(0)
    # print('obs.shape: ', obs.shape)                # torch.Size([1, 6, 34, 4])
    # print('legactions.shape: ', legactions.shape)  # torch.Size([1, 38])
    res, value = net(obs, legactions)
    pi = F.softmax(res, dim=-1)
    act = torch.argmax(pi, dim=-1, keepdim=True).item()
    action[0][act] = 1

    return action

def randomAction():
    action = [[0] * 38]
    act = random.choice(list(range(38)))
    action[0][act] = 1

    return action

def main():
    game = make('chessandcard-mahjong_v3')

    all_observes = game.all_observes
    while not game.is_terminal():
        # step = "step%d" % game.step_cnt
        # if game.step_cnt % 10 == 0:
        #     print(step)
        
        joint_act = []
        joint_act.append(getAction(all_observes[0]))
        for i in range(1, 4):
            joint_act.append(randomAction())
        all_observes, reward, done, info_before, info_after = game.step(joint_act)

    # print('reward: ', reward)
    return [reward['player_0'], reward['player_1'], reward['player_2'], reward['player_3']]


if __name__ == "__main__":
    device = torch.device("cpu")
    net = Resnet18().to(device)
    net = torch.nn.DataParallel(net)
    elo_game_times = 1000
    st = time.time()
    SumWriter = SummaryWriter(log_dir="./save/tb")

    while True:
    # for _ in range(1):
        win_time, score = 0, 0
        load(net)

        # for i in tqdm(range(elo_game_times)):
        for i in range(elo_game_times):
            res = main()
            score += res[0]
            win_time += res[0] if res[0] > 0 else 0

        
        xlable = (time.time()-st) // 60
        test_win = win_time / (elo_game_times)
        test_score = score / (elo_game_times)
        SumWriter.add_scalar('win rate elo / min', test_win, xlable)
        SumWriter.add_scalar('score elo / min', test_score, xlable)

        print("win rate elo: {}, score elo: {}".format(test_win, test_score))
        print("time use {:.2f}s: ".format(time.time()-st))
        print("***"*10)
