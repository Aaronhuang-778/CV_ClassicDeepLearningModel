# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/8/28 15:52
import torch
from torch import nn


class ConvC2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5):
        super(ConvC2, self).__init__()
        # 定义特征图的映射方式
        map = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        mapInfo = torch.tensor(map, dtype=torch.long)
        self.register_buffer("mapInfo", mapInfo)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.convs = {}

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