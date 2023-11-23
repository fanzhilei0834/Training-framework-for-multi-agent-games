# AAMAS2023 竞赛桥牌赛道 RL 自博弈 learner 部分
# author: fanzl
# date: 2023/4/14
# version: 1.1
# 版本改进：
# 待完善:


import time
import os
import zmq
import pickle
import torch
import torch.nn.functional as F
from pyarrow import deserialize
import numpy as np
from multiprocessing import Process
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from model import MLPnet, orthogonal_init
from mempool import MemPoolManager, MultiprocessingMemPool


class Learner:
    def __init__(self, net_lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MLPnet().to(self.device)
        self.net.apply(orthogonal_init)
        # self.net = torch.nn.DataParallel(self.net)
        self.net.train()  # 训练模式，注意有BN层
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=net_lr)
        self.batch_size = 4096
        self.policy_clip_range = 0.5
        self.value_clip_range = 0.5
        self.SumWriter = SummaryWriter(log_dir="./save/tb")
        self.tb_x = 0
    
    def _calc_policy_loss(self, sample_action, old_prob, prob, sampled_advantage):
        prob_action = torch.gather(prob, dim=1, index=sample_action)
        prob_action_old = old_prob
        ratio = prob_action / prob_action_old
        # self.SumWriter.add_scalar('mean ratio', torch.mean(ratio).item(), self.tb_x)
        # self.SumWriter.add_scalar('max ratio', max(ratio).item(), self.tb_x)
        # self.SumWriter.add_scalar('min ratio', min(ratio).item(), self.tb_x)
        self.SumWriter.add_scalar('up clip rate', sum(ratio > 1 + self.policy_clip_range).item()/self.batch_size, self.tb_x)
        self.SumWriter.add_scalar('down clip rate', sum(ratio < 1 - self.policy_clip_range).item()/self.batch_size, self.tb_x)
        # self.SumWriter.add_scalar('mean prob', torch.mean(prob.gather(1, sample_action)).item(), self.tb_x)
        clipped_ratio = torch.clamp(ratio, 0.0, 3.0)
        norm_adv = sampled_advantage
        surr1 = clipped_ratio * norm_adv
        surr2 = torch.clamp(ratio, 1.0 - self.policy_clip_range, 1.0 + self.policy_clip_range) * norm_adv
        policy_loss = torch.mean(torch.min(surr1, surr2))
        return policy_loss
    
    def _calc_value_loss(self, value, old_value, sampled_advantage):
        clipped_value = old_value + torch.clamp(value - old_value, - self.value_clip_range, self.value_clip_range)
        sampled_reward_sum = old_value + sampled_advantage
        value_loss = 0.5 * torch.mean(
            torch.max(torch.pow(value - sampled_reward_sum, 2), torch.pow(clipped_value - sampled_reward_sum, 2)))
        
        return value_loss
    
    def _calc_entropy_loss(self, pi_logits):
        logits = pi_logits - torch.max(pi_logits, dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(logits)
        exp_logits_sum = torch.sum(exp_logits, dim = -1, keepdim = True)
        p = exp_logits / exp_logits_sum
        temp_entropy_loss = torch.sum(p * (torch.log(exp_logits_sum) - logits), dim = -1)
        entropy_loss = torch.mean(temp_entropy_loss)
        return entropy_loss

    def update(self, transition_dict):
        self.optimizer.zero_grad()
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).long().view(-1, 1).to(self.device)
        advantages = torch.tensor(transition_dict["advantages"], dtype=torch.float).view(-1, 1).to(self.device)
        probs_old = torch.tensor(transition_dict["probs_old"], dtype=torch.float).view(-1, 1).to(self.device)
        value_old = torch.tensor(transition_dict["value_old"], dtype=torch.float).view(-1, 1).to(self.device)
        legal_actions = torch.tensor(transition_dict["legal_actions"], dtype=torch.float).to(self.device)

        res, value = self.net(states, legal_actions)
        pi = F.softmax(res, dim=-1)
            
        policy_loss = self._calc_policy_loss(actions, probs_old, pi, advantages)
        value_loss = self._calc_value_loss(value, value_old, advantages)
        entropy_loss = self._calc_entropy_loss(res)

        loss = -(policy_loss - 0.5 * value_loss + 0.1 * entropy_loss)
        loss.backward()
        
        self.optimizer.step()

        self.SumWriter.add_scalar('policy_loss', policy_loss.item(), self.tb_x)
        self.SumWriter.add_scalar('entropy_loss', entropy_loss.item(), self.tb_x)
        self.SumWriter.add_scalar('value_loss', value_loss.item(), self.tb_x)
        self.SumWriter.add_scalar('advantage', advantages.mean().item(), self.tb_x)

        self.tb_x += 1
        print("    policy loss: {} , value loss: {} , entropy loss: {}".format(policy_loss.item(), value_loss.item(), entropy_loss.item()))

    def dump_weight(self):
        weight_dir = "./save/weight.pkl"
        self.net.to('cpu')

        # dump net weight
        while True:
            try:
                with open(weight_dir, "wb") as fwei:
                    pickle.dump(self.net.state_dict(), fwei)
                # print("\tResnet dump weight succeed")
                break
            except Exception as err:
                print("\t*** Resnet dump weight error! redump soon. ***")
                print(err)
                time.sleep(0.1)
        self.net.to(self.device)


def main():
    def recv_data(mem_pool):
        print("ZMQ start, mode REP")
        context = zmq.Context()
        data_socket = context.socket(zmq.REP)
        data_socket.bind(f'tcp://*:5566')

        while True:
            data = deserialize(data_socket.recv())

            while True:
                try:
                    mem_pool.push(data)
                    break
                except:
                    with open('error_data.pkl', 'wb') as ferr:
                        pickle.dump(data, ferr)
                        print("mem_pool.push error !!")
                        time.sleep(1)
            data_socket.send(b"message received")
    
    learner = Learner(net_lr=2.5e-4)
    learner.dump_weight()

    # 开启 memory pool
    manager = MemPoolManager()
    manager.start()
    mem_pool = manager.MemPool(5000)
    Process(target=recv_data, args=(mem_pool,)).start()

    # 开启 memory pool 吞吐量监控
    Process(target=mem_pool.record_throughput, args=(600,)).start()

    start_time = time.time()
    while True:
        # print("    mem_pool.__len__: ", mem_pool.__len__())
        time.sleep(1)

        if mem_pool.__len__() >= learner.batch_size:
            transition_dict = mem_pool.sample(learner.batch_size)
            print("\tupdate time: ", time.time() - start_time)
            learner.update(transition_dict)
            learner.dump_weight()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main()