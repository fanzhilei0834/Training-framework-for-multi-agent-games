import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPnet(nn.Module):
    '''
    AAMAS2023 桥牌赛道 MLP网络
    '''
    def __init__(self, state_dim=573, action_dim=91):
        super(MLPnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 1024)
        self.fc4 = torch.nn.Linear(1024, 1024)
        self.fc5 = torch.nn.Linear(1024, 128)
        self.fc6 = torch.nn.Linear(1024, 512)
        self.L = 10**10
        self.policy = torch.nn.Linear(512, action_dim)
        self.value = torch.nn.Linear(128, 1)
    
    def forward(self, x, legalact):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        _value = F.relu(self.fc5(x))
        value = self.value(_value)
        _policy = F.relu(self.fc6(x))
        _policy = self.policy(_policy)
        policy = _policy - (1 - legalact) * self.L

        return policy, value


def orthogonal_init(layer, gain=0.1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        # nn.init.constant_(layer.bias, 0)

def load(net, weight_dir="./save/weight.pkl"):
    '''
    加载新模型
    '''
    # load net weight
    while True:
        try:
            with open(weight_dir, "rb") as fwei:
                # net = torch.nn.DataParallel(net)
                net_weight = pickle.load(fwei)
                net.load_state_dict(net_weight)
                net.eval()
            # print("Resnet load weight succeed")
            break
        except Exception as err:
            # print("*** Resnet load weight error! reload soon. ***")
            # print('\n', '==='*20)
            # print(err)
            # print('==='*20, '\n')
            time.sleep(2)

def convert(observation):
    return torch.tensor(observation, dtype=torch.float)
    
def test():
    import pickle
    import numpy as np
    data = torch.rand((100, 145, 4, 9), dtype=torch.float)
    with open('tensor.pkl', 'wb') as f:
        pickle.dump(data, f)
        print(type(data))
    nparray = data.numpy()
    with open("nparray.pkl", "wb") as f:
        pickle.dump(nparray, f)
        print(type(nparray))
    li = data.tolist()
    with open("list.pkl", "wb") as f:
        pickle.dump(li, f)
        print(type(li))


if __name__ == "__main__":
    test()