import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        #shortcut
        self.shortcut = nn.Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet_value(nn.Module):
    def __init__(self, block, num_block, num_classes=38):
        super().__init__()
        self.conv_pre = nn.Conv2d(6, 128, kernel_size=3, padding=1, bias=False)
        self.in_channels = 64
        self.L = 10**10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block, 1)
        #self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        #self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        #self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*4*34, num_classes)
        self.fc_value1 = nn.Linear(64*4*34, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, legalact):
        output = self.conv_pre(x)
        output = self.conv1(output)
        output = self.conv2_x(output)
        #output = self.conv3_x(output)
        #output = self.conv4_x(output)
        #output = self.conv5_x(output)
        #output = self.avg_pool(output)

        output = output.view(output.size(0), -1)
        value = F.relu(self.fc_value1(output))
        value = self.fc_value2(value)
        output = self.fc(output)

        res = output - (1 - legalact) * self.L

        return res, value


class MLPnet(nn.Module):
    '''
    AAMAS2023 桥牌赛道 MLP网络
    '''
    def __init__(self, state_dim=204, action_dim=38):
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


def Resnet18():
    return ResNet_value(BasicBlock, 9)


def Resnet34():
    return ResNet_value(BasicBlock, 16)


def Resnet50():
    return ResNet_value(BottleNeck, 16)


def convert(observation, player_index):
    # channel 0
    channel0 = torch.tensor(observation[0], dtype=torch.float).unsqueeze(0)
    # channel 1
    channel1 = torch.tensor(observation[1], dtype=torch.float).unsqueeze(0)
    # 己方
    channel2 = torch.tensor(observation[player_index+2], dtype=torch.float).unsqueeze(0)
    # 下家
    channel3 = torch.tensor(observation[(player_index + 1) % 4 + 2], dtype=torch.float).unsqueeze(0)
    # 对家
    channel4 = torch.tensor(observation[(player_index + 2) % 4 + 2], dtype=torch.float).unsqueeze(0)
    # 上家
    channel5 = torch.tensor(observation[(player_index + 3) % 4 + 2], dtype=torch.float).unsqueeze(0)

    return torch.cat((channel0, channel1, channel2, channel3, channel4, channel5), 0)

def convert2(observation, player_index):
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


def convert3(observation, player_index):
    observation = torch.tensor(observation)
    part0  = torch.sum(observation[0], dim=1)
    part1  = torch.sum(observation[1], dim=1)
    part2  = torch.sum(observation[player_index+2], dim=1)
    part3  = torch.sum(observation[(player_index + 1) % 4 + 2], dim=1)
    part4  = torch.sum(observation[(player_index + 2) % 4 + 2], dim=1)
    part5  = torch.sum(observation[(player_index + 3) % 4 + 2], dim=1)

    _obs = torch.cat((part0, part1, part2, part3, part4, part5), 0)
    obs = torch.tensor(_obs, dtype=torch.float)

    return obs


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
    # import pickle
    # def orthogonal_init(layer, gain=0.1):
    #     if isinstance(layer, (nn.Conv2d, nn.Linear)):
    #         nn.init.orthogonal_(layer.weight, gain=gain)
    #         # nn.init.constant_(layer.bias, 0)

    # actor = ResNet_policy()
    # critic = ResNet_value()
    # actor.apply(orthogonal_init)
    # critic.apply(orthogonal_init)
    # with open("./save/policy.pkl", "wb") as f:
    #     pickle.dump(actor.state_dict(), f)
    # with open("./save/value.pkl", "wb") as f:
    #     pickle.dump(critic.state_dict(), f)
    test()