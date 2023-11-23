#!/usr/bin/python
#coding:utf-8

# Resnet model
# author: fanzl
# version: 1.1
# time: 2023/04/12

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels !=  out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Resnet18(nn.Module):
    def __init__(self, block=BasicBlock, num_block=9, num_classes=38):
        super().__init__()
        self.conv_pre = nn.Conv2d(6, 128, kernel_size=3, padding=1, bias=False)
        self.in_channels = 64
        self.L = 100000

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block, 1)
        self.fc = nn.Linear(8704, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x, legalact):
        # print('state shape in model: ', x.shape)
        output = self.conv_pre(x)
        output = self.conv1(output)
        output = self.conv2_x(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        res = output - (1 - legalact) * self.L

        return res