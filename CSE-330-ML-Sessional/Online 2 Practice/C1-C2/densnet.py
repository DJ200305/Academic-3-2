import torch
import torch.nn as nn
import torch.functional as F
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, 4 * growth_rate, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], dim=1)
        return out
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_ch, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_ch + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
class Transition(nn.Module):
    def __init__(self, in_ch, compression=0.5):
        super().__init__()
        out_ch = int(in_ch * compression)

        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x
class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 2 * growth_rate, 3, padding=1, bias=False)

        channels = 2 * growth_rate

        self.db1 = DenseBlock(6, channels, growth_rate)
        channels += 6 * growth_rate
        self.tr1 = Transition(channels)
        channels = int(channels * 0.5)

        self.db2 = DenseBlock(12, channels, growth_rate)
        channels += 12 * growth_rate
        self.tr2 = Transition(channels)
        channels = int(channels * 0.5)

        self.db3 = DenseBlock(24, channels, growth_rate)
        channels += 24 * growth_rate
        self.tr3 = Transition(channels)
        channels = int(channels * 0.5)

        self.db4 = DenseBlock(16, channels, growth_rate)
        channels += 16 * growth_rate

        self.bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.tr1(self.db1(x))
        x = self.tr2(self.db2(x))
        x = self.tr3(self.db3(x))
        x = self.db4(x)

        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
