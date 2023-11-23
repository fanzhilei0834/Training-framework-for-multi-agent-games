# AAMAS2023 竞赛桥牌赛道 RL 自博弈 localgame 部分
# author: fanzl
# date: 2023/4/14
# version: 1.1
# 版本改进：
# 待完善:


import time
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

from env.chooseenv import make
from model import MLPnet, load, convert
from agent.RLv1.submission import my_controller as RLv1
# from agent.RLv2.submission import my_controller as RLv2


def getAction(observation):
    global net
    action = [[0] * 91]

    if not observation['obs']:
        action[0][-1] = 1
        return action
    
    obs = convert(observation['obs']['observation']).unsqueeze(0)
    legactions = torch.tensor(observation['obs']['action_mask'], dtype=torch.float).unsqueeze(0)

    res, value = net(obs, legactions)
    pi = F.softmax(res, dim=-1)
    act = torch.argmax(pi, dim=-1, keepdim=True).item()
    action[0][act] = 1

    return action

def oppPolicy(observation):
    return RLv1(observation)

def randomAction():
    action = [[0] * 91]
    act = random.choice(list(range(91)))
    action[0][act] = 1

    return action

def main():
    game = make('bridge')

    all_observes = game.all_observes
    while not game.is_terminal():
        joint_act = []
        joint_act.append(getAction(all_observes[0]))
        joint_act.append(oppPolicy(all_observes[1]))
        joint_act.append(getAction(all_observes[2]))
        joint_act.append(oppPolicy(all_observes[3]))
        all_observes, reward, done, info_before, info_after = game.step(joint_act)

    return [reward["player_0"], reward["player_1"], reward["player_2"], reward["player_3"]]


if __name__ == "__main__":
    net = MLPnet()
    elo_game_times = 1000
    st = time.time()
    SumWriter = SummaryWriter(log_dir="./save/tb")

    while True:
        win_time, score = [0, 0, 0, 0], [0, 0, 0, 0]
        load(net)

        for i in tqdm(range(elo_game_times)):
            res = main()
            for j in range(4):
                win_time[j] += 1 if res[j] > res[j-1] else 0
                score[j] += res[j]

        xlable = (time.time()-st) // 60
        test_win = win_time[0] / (elo_game_times)
        test_score = score[0] / (elo_game_times)
        SumWriter.add_scalar('win rate elo / min', test_win, xlable)
        SumWriter.add_scalar('score elo / min', test_score, xlable)

        print("win times elo: {}, score elo: {}".format(win_time, score))
        print("time use {:.2f}s: ".format(time.time()-st))
        print("***"*10)
