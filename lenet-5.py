# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/8/28 15:32
import torch
from torch import nn, optim

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


class MapConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5):
        super(MapConv, self).__init__()
        # 定义特征图的映射方式
        mapInfo = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        mapInfo = torch.tensor(mapInfo, dtype=torch.long)
        self.register_buffer("mapInfo", mapInfo)  # 在Module中的buffer中的参数是不会被求梯度的
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.convs = {}  # 将每一个定义的卷积层都放进这个字典

        # 对每一个新建立的卷积层都进行注册，使其真正成为模块并且方便调用
        for i in range(self.out_channel):
            conv = nn.Conv2d(mapInfo[:, i].sum().item(), 1, kernel_size)
            convName = "conv{}".format(i)
            self.convs[convName] = conv
            self.add_module(convName, conv)

    def forward(self, x):
        outs = []  # 对每一个卷积层通过映射来计算卷积，结果存储在这里
        for i in range(self.out_channel):
            mapIdx = self.mapInfo[:, i].nonzero().squeeze()
            convInput = x.index_select(1, mapIdx)
            convOutput = self.convs['conv{}'.format(i)](convInput)
            outs.append(convOutput)
        return torch.cat(outs, dim=1)

