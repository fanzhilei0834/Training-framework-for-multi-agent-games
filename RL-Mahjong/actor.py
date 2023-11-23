# AAMAS2023 竞赛四人麻将赛道 RL 自博弈 actor 部分
# author: fanzl
# date: 2023/4/14
# version: 1.1
# 版本改进：
# 待完善:


import random
import time
import zmq
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from pyarrow import serialize

from env.chooseenv import make
from mempool import ActorBuffer
from ResnetModel import Resnet18, load, convert


def getAction(observation):
    global net
    global data_buffer
    action = [[0] * 38]

    if not observation['obs']:
        action[0][-1] = 1
        return action
    
    obs = convert(observation['obs']['observation'], observation['controlled_player_index']).unsqueeze(0)
    legactions = torch.tensor(observation['obs']['action_mask'], dtype=torch.float).unsqueeze(0)
    # print('obs.shape: ', obs.shape)                # torch.Size([1, 6, 34, 4])
    # print('legactions.shape: ', legactions.shape)  # torch.Size([1, 38])
    res, value = net(obs, legactions)
    pi = F.softmax(res, dim=-1)
    act = torch.distributions.Categorical(pi).sample().item()
    prob = pi[0][act].item()
    action[0][act] = 1

    # for i in range(38):
    #     if legactions[0][i] == 1:
    #         print(pi[0][i].item(), end=' ')
    # print(' ')

    assert 0 <= observation['controlled_player_index'] <= 3
    data_buffer[observation['controlled_player_index']].obs_buf.extend(obs.tolist())
    data_buffer[observation['controlled_player_index']].act_buf.append(act)
    data_buffer[observation['controlled_player_index']].rew_buf.append(0.)
    data_buffer[observation['controlled_player_index']].val_buf.append(value.item())
    data_buffer[observation['controlled_player_index']].prob_buf.append(prob)
    data_buffer[observation['controlled_player_index']].legact_buf.extend(legactions.tolist())
    data_buffer[observation['controlled_player_index']].data_n += 1

    return action

def send_data(score_ls):
    global socket
    global data_buffer
    data = {
        "states": [],
        "actions": [],
        "advantages": [],
        "probs_old": [],
        "value_old": [],
        "legal_actions": [],
        "rewards": [],
        "info": [],
        "traj_lenth": 0,
        "score": max(score_ls)
    }

    for i in range(4):
        if score_ls[i] == 1:
            score_ls[i] = 3
        # elif score_ls[i] == 0:  # 流局惩罚
        #     score_ls[i] = -1

    for i in range(4):
        buffer = data_buffer[i]
        tmp = buffer.finish_path(score_ls[i])
        if tmp:
            data["states"].extend(tmp["states"])
            data["actions"].extend(tmp["actions"])
            data["advantages"].extend(tmp["advantages"])
            data["probs_old"].extend(tmp["probs_old"])
            data["value_old"].extend(tmp["value_old"])
            data["legal_actions"].extend(tmp["legal_actions"])
            data["rewards"].extend(tmp["rewards"])
            data["traj_lenth"] += tmp["traj_lenth"]

    data["states"] = np.array(data["states"])
    data["actions"] = np.array(data["actions"])
    data["advantages"] = np.array(data["advantages"])
    data["probs_old"] = np.array(data["probs_old"])
    data["value_old"] = np.array(data["value_old"])
    data["legal_actions"] = np.array(data["legal_actions"])
    data["rewards"] = np.array(data["rewards"])

    # print(score_ls)
    # print(data["advantages"])
    # print(data["value_old"])
    # print("actions:\n", data["actions"])
    # print("rewards:\n", data["rewards"])

    _data = serialize(data)
    data = _data.to_buffer()
    socket.send(data)
    message = socket.recv()

    for i in range(4):
        data_buffer[i].clear()

def main():
    game = make('chessandcard-mahjong_v3')

    all_observes = game.all_observes
    while not game.is_terminal():
        # step = "step%d" % game.step_cnt
        # if game.step_cnt % 10 == 0:
        #     print(step)
        
        joint_act = []
        for it in all_observes:
            joint_act.append(getAction(it))
        all_observes, reward, done, info_before, info_after = game.step(joint_act)

    # print('reward: ', reward)
    send_data([reward['player_0'], reward['player_1'], reward['player_2'], reward['player_3']])


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cpu")
    net = Resnet18().to(device)
    net = torch.nn.DataParallel(net)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://172.17.0.3:5566")
    print("zmq connected, use pyarrow")

    data_buffer = [ActorBuffer(), ActorBuffer(), ActorBuffer(), ActorBuffer()]

    while True:
        # try: 
        #     load(net)
        #     print(main())
        # except Exception as err:
        #     print(err)
        #     time.sleep(1)
        load(net)
        main()