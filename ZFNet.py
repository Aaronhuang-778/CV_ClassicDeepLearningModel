# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/8/28 17:13
import torch
import torch.nn as nn


class ZFNet(nn.Module):
    def __init__(self):
        super(ZFNet, self).__init__()
        self.conv = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 第二次
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 第三层
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第四层
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第五层
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        # print(feature.shape)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
