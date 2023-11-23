# AAMAS2023 竞赛桥牌赛道 RL 自博弈 debug 部分
# author: fanzl
# date: 2023/5/8
# version: 1.1
# 版本改进：
# 待完善:


import time
import random
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from env.chooseenv import make
# from agent.RLv1.submission import my_controller as RLv1
from agent.RLv2.submission import my_controller as RLv2
from agent.RLv3.submission import my_controller as RLv3


def getAction1(observation):
    return RLv3(observation)

def getAction2(observation):
    return RLv2(observation)

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
        joint_act.append(getAction1(all_observes[0]))
        joint_act.append(getAction2(all_observes[1]))
        joint_act.append(getAction1(all_observes[2]))
        joint_act.append(getAction2(all_observes[3]))
        all_observes, reward, done, info_before, info_after = game.step(joint_act)

    return [reward["player_0"], reward["player_1"], reward["player_2"], reward["player_3"]]


if __name__ == "__main__":
    st = time.time()
    win_time, score = [0, 0, 0, 0], [0, 0, 0, 0]

    for i in tqdm(range(10000)):
        res = main()
        for j in range(4):
            win_time[j] += 1 if res[j] > res[j-1] else 0
            score[j] += res[j]

    print("win times elo: {}, score elo: {}".format(win_time, score))
    print("time use {:.2f}s: ".format(time.time()-st))
    print("***"*10)
