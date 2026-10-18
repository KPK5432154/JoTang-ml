
import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积层、激活函数、池化层
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 将每个通道的特征图压缩成一个数值,64特征图
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 输出一个数值，用于区分猫和狗
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)#合并
        x = self.classifier(x)
        return x
