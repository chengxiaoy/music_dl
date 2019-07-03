from torch.nn import Module
from torch import nn
import torch
from util.audio_processor import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN_Baseline(Module):
    """
    model from Music Genre Recognition using Deep Neural Networks and Transfer Learning
    maybe not work
    <<Music Genre Recognition using Deep Neural Networks and Transfer Learning>>

    """

    def __init__(self):
        super(CNN_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, (96, 3))
        self.maxpool = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(128, 64, (1, 3))
        self.globalmax = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.globalmax(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    base_model = CNN_Baseline().to(device)
    audio_path = "../audio/bensound-actionable.mp3"
    mel_spectrum = compute_melgram(audio_path)
    mel_spectrum = torch.from_numpy(mel_spectrum).float()

    output = base_model(mel_spectrum)
    print(output.shape)
