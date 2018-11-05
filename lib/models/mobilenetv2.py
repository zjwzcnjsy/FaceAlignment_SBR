import torch.nn as nn
import torch.nn.init as init
from copy import deepcopy


def conv_bn_relu(inp, oup, ks=3, stride=1, pad=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, ks, stride, pad, bias=bias),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=False)
    )


def dw_conv_bn_relu(inp, oup, ks=3, stride=1, pad=1, bias=False):
    assert inp == oup
    return nn.Sequential(
        nn.Conv2d(inp, oup, ks, stride, pad, groups=inp, bias=bias),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=False)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert self.stride in (1, 2)

        self.use_skip = (self.stride == 1 and inp == oup)

        self.block = nn.Sequential(
            # pw
            conv_bn_relu(inp, inp * t, 1, 1, 0, bias=False),
            # dw
            dw_conv_bn_relu(inp * t, inp * t, 3, self.stride, 1, bias=False),
            # pw-linear
            nn.Conv2d(inp * t, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_skip:
            return x + self.block(x)
        else:
            return self.block(x)


class FirstInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t):
        super(FirstInvertedResidual, self).__init__()
        self.stride = stride
        assert self.stride in (1, 2)

        self.block = nn.Sequential(
            # pw
            conv_bn_relu(inp, inp * t, 1, 1, 0, bias=False),
            # dw
            dw_conv_bn_relu(inp * t, inp * t, 3, self.stride, 1, bias=False),
            # pw-linear
            nn.Conv2d(inp * t, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.block(x)


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class MobileNetV2(nn.Module):
    def __init__(self, config, pts_num):
        super(MobileNetV2, self).__init__()

        self.config = deepcopy(config)
        self.downsample = 8
        self.pts_num = pts_num
        # setting of inverted residual blocks
        self.inverted_residual_setting = [
            # t, c, n, s
            [2, 24, 2, 2],
            [2, 32, 3, 2],
            [2, 64, 4, 2],
        ]
        input_channel = 32
        self.features = [conv_bn_relu(3, input_channel, 3, 2, 1, bias=False),
                         FirstInvertedResidual(input_channel, 16, 1, 1)]
        input_channel = 16
        for t, c, n, s in self.inverted_residual_setting:
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(input_channel, c, s, t))
                else:
                    self.features.append(
                        InvertedResidual(input_channel, c, 1, t))
                input_channel = c
        input_channel = 64
        self.features = nn.Sequential(*self.features)
        self.fc_landmark68 = nn.Linear(input_channel * 4 * 4, 2*self.pts_num)
        self.apply(weights_init)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        landmark68 = self.fc_landmark68(x)
        return landmark68.view(-1, self.pts_num, 2)
