'''
Author: yuqi
Date: 2021-04-26 10:12:54
LastEditTime: 2021-04-26 17:55:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /tsn-pytorch/model_zoo/alm.py
'''
from torch import nn

from transforms import *
from torch.nn.init import normal, constant

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# Additive Long-range module
# Use 1*k*k -> t*1*1 instead of 3d cnn
class ALM(nn.Module):
    def __init__(self, num_classes, num_segments, 
                 inplanes, 
                 modality='RGB', dropout=0.8):
        super(ALM, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        
        planes = 512
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv3d_spatial = nn.Conv3d(planes, planes, kernel_size=(1,3,3))
        self.conv3d_temporal = nn.Conv3d(planes, planes, kernel_size=(3,1,1))
        self.bn3d = nn.BatchNorm3d(planes)
        # self.conv3 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout3d(dropout)
        self.fc = nn.Linear(planes, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # input size here is (batch_size, N_seg, fc_input, 7, 7), fc_input here is 2048
        # and we reshape it to (batch_size*N_seg, fc_input, 7, 7)
        # print('input size: ', input.shape)
        fc_input = 2048
        # print('fc_input: ', fc_input)
        input = input.view((-1, fc_input) + input.size()[-2:])
        # print(input.shape)
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        # shape here should be (batch_size*N_seg, planes, 7, 7)
        # then we reshape to (batch_size, C_in, D, 7, 7) and feed to 3D, 
        # where C_in=planes and D=N_seg
        output = output.view((-1, self.num_segments) + output.size()[1:]).transpose(1,2)     # ()
        output = self.conv3d_spatial(output)
        output = self.conv3d_temporal(output)
        output = self.bn3d(output)
        output = self.relu(output)

        output = self.avgpool(output)
        # shape here should be (batch_size, planes, 1, 1, 1)
        output = torch.flatten(output, 1)
        output = self.drop(output)
        output = self.fc(output)
        # print('3d output: ', output.shape)
        # output = self.softmax(output)
        return output