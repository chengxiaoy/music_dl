from torch.nn import Module
from torch import nn


class CNN_Choi(Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 64, 3)

        self.bn1 = nn.BatchNorm2d()

        x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
