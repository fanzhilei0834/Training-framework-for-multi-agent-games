# memory pool

import random
import time
import numpy as np
from multiprocessing.managers import BaseManager
from collections import deque
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


class ActorBuffer:

    def __init__(self, gamma=0.99, lam=0.95):
        self.obs_buf = []
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.val_buf = []
        self.prob_buf = []
        self.legact_buf = []
        self.data_n = 0
        self.gamma, self.lam = gamma, lam

    
    def clear(self):
        self.obs_buf = []
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.val_buf = []
        self.prob_buf = []
        self.legact_buf = []
        self.data_n = 0
    
    def finish_path(self, score):
        if not self.rew_buf:
            return {}

        last_value = score
        last_adv = 0
        for i in range(len(self.val_buf)-1, -1, -1):
            delta = self.rew_buf[i] + self.gamma * last_value - self.val_buf[i]
            last_adv = delta + self.gamma * self.lam * last_adv
            self.adv_buf.insert(0, last_adv)
            last_value = self.val_buf[i]

        data = {
            "states": self.obs_buf,
            "actions": self.act_buf,
            "advantages": self.adv_buf,
            "probs_old": self.prob_buf,
            "value_old": self.val_buf,
            "legal_actions": self.legact_buf,
            "rewards": self.rew_buf,
            "traj_lenth": self.data_n
        }
        assert len(data["states"]) == len(data["actions"]) == len(data["advantages"]) == len(data["probs_old"]) == len(data["value_old"]) == len(data["legal_actions"]) == len(data["rewards"])

        return data

class MultiprocessingMemPool():
    def __init__(self, capacity: int = None):
        # super().__init__()
        self.capacity = capacity
        self.data = deque(maxlen = self.capacity)
        self._receiving_data_throughput = 0    # 用于接收速率统计
        self._consuming_data_throughput = 0    # 用于消耗速率统计
        self.SumWriter = SummaryWriter(log_dir="./save/tb")
        self.data_n = 0
        self.start_time = time.time()
        self.win100 = deque(maxlen=100)
    
    def push(self, data):
        if not data:
            return
        states = data["states"]
        actions = data["actions"]
        advantages = data['advantages']
        probs_old = data['probs_old']
        value_old = data["value_old"]
        legal_actions = data["legal_actions"]
        time_tmp = time.time()-self.start_time
        tmp = list(zip(states, actions, advantages, probs_old, value_old, legal_actions))
        self.win100.append(1 if data["score"] > 0 else 0)
        # self.SumWriter.add_scalar('score', data["score"], time_tmp)
        # self.SumWriter.add_scalar('win rate deque', sum(self.win100), time_tmp)
        print("\tpush: ", len(tmp), "time: ", round(time_tmp,3), "score: ", data["score"], "win rate: ", sum(self.win100))
        # self.SumWriter.add_scalar('step per game', data["traj_lenth"], time_tmp)
        
        # print("push: ", len(tmp), "time: ", time_tmp)
        self.data.extend(tmp)
        self.data_n = min(self.data_n + len(tmp), self.capacity)
        self._receiving_data_throughput += len(tmp)  
        assert len(self.data) == self.data_n
    
    def sample(self, size):
        tmp = random.sample(self.data, size)
        states, actions, advantages, probs_old, value_old, legal_actions = zip(*tmp)
        transition_dict = {
            "states": np.array(states),
            "actions": np.array(actions),
            "advantages": np.array(advantages),
            "probs_old" : np.array(probs_old),
            "value_old" : np.array(value_old),
            "legal_actions": np.array(legal_actions)
        }
        self._consuming_data_throughput += size
        # self._clear()

        return transition_dict
    
    def _clear(self):
        self.data = deque(maxlen = self.capacity)
        self.data_n = 0
        self._receiving_data_throughput = 0
        self._consuming_data_throughput = 0
    
    def __len__(self):
        return len(self.data)
    
    # def __del__(self):
    #     self.fcsv.close()

    def _data_n(self):
        return self.data_n
    
    def _get_receiving_data_throughput(self):
        return self._receiving_data_throughput

    def _get_consuming_data_throughput(self):
        return self._consuming_data_throughput

    def _reset_receiving_data_throughput(self):
        self._receiving_data_throughput = 0

    def _reset_consuming_data_throughput(self):
        self._consuming_data_throughput = 0
    
    def _tb(self, title, y, x):
        self.SumWriter.add_scalar(title, y, x)
    

    # @classmethod
    def record_throughput(self, interval=10):
        """Print receiving and consuming periodically"""
        start_time = time.time()

        while True:
            self._reset_receiving_data_throughput()
            self._reset_consuming_data_throughput()

            time.sleep(interval)
            Rfps = self._get_receiving_data_throughput() / interval
            Cfps = self._get_consuming_data_throughput() / interval

            print(f'Receiving FPS: {Rfps:.2f}, '
                  f'Consuming FPS: {Cfps:.2f}')
            tmp_t = time.time()-start_time
            self._tb('Receiving FPS', Rfps, tmp_t)
            self._tb('Consuming FPS', Cfps, tmp_t)

class MemPoolManager(BaseManager):
    pass

MemPoolManager.register('MemPool', MultiprocessingMemPool,
                        exposed=['__len__', 'push', 'sample', 'clear', '_data_n', '_tb'
                        '_get_receiving_data_throughput','_get_consuming_data_throughput', 
                        '_reset_receiving_data_throughput','_reset_consuming_data_throughput',
                        'record_throughput'])
