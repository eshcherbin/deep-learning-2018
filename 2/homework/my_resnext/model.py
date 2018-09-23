import torch
import torch.nn as nn


class ResNeXtBottleneckC(nn.Module):
    """
    ResNeXt bottleneck block type C
    as described in https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, in_channels, out_channels, cardinality, stride=1):
        super().__init__()
        mid_channels = cardinality * (in_channels / 64) * 4

        self.conv1 = nn.Conv2d(in_channels,
                               mid_channels,
                               kernel_size=1,
                               stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels,
                               mid_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels,
                               out_channels,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=stride,
                                                    bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        result = self.conv1(x)
        result = self.bn1(result)
        result = self.relu(result)

        result = self.conv2(result)
        result = self.bn2(result)
        result = self.relu(result)

        result = self.conv3(result)
        result = self.bn3(result)
        result = self.relu(result)

        shortcut = x if self.shortcut is None else self.shortcut(x)
        result += shortcut
        result = self.relu(result)

        return result


class MyResNeXt(nn.Module):
    """
    ResNeXt model (ImageNet architecture)
    as described in https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, layers, cardinality, n_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(cardinality, 64, layers[0])
        self.layer2 = self._make_layer(cardinality, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(cardinality, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(cardinality, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, n_classes)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _make_layer(cardinality, channels, n_blocks, stride=1):
        return nn.Sequential(*(ResNeXtBottleneckC(channels,
                                                  channels * 4,
                                                  cardinality,
                                                  stride=stride)
                               for _ in range(n_blocks)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext50(pretrained_path=None, **kwargs):
    model = MyResNeXt([3, 4, 6, 3], 32, **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    return model


def resnext101(pretrained_path=None, **kwargs):
    model = MyResNeXt([3, 4, 23, 3], 32, **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    return model


def resnext152(pretrained_path=None, **kwargs):
    model = MyResNeXt([3, 8, 36, 3], 32, **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    return model
