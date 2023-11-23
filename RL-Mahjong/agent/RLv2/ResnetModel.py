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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        #shortcut
        self.shortcut = nn.Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
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
            nn.BatchNorm2d(64),
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


def Resnet18():
    return ResNet_value(BasicBlock, 9)


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