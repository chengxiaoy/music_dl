from torch.nn import Module
from torch import nn
import torch
from torchvision.models import resnet34
from util.audio_processor import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN_Choi(Module):
    """
    fcn6 from AUTOMATIC TAGGING USING DEEP CONVOLUTIONAL NEURAL NETWORKS
    """

    def __init__(self):
        super(CNN_Choi, self).__init__()
        # block1
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.activate1 = nn.ELU()
        self.maxpool1 = nn.MaxPool2d((2, 4))

        # block2
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.activate2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d((2, 4))

        # block3
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.activate3 = nn.ELU()
        self.maxpool3 = nn.MaxPool2d((2, 4))

        # block4
        self.conv4 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.activate4 = nn.ELU()
        self.maxpool4 = nn.MaxPool2d((3, 5))

        # block5
        self.conv5 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(2048)
        self.activate5 = nn.ELU()
        self.maxpool5 = nn.MaxPool2d((4, 4))

        # block6
        self.conv6 = nn.Conv2d(2048, 1024, 1)

        self.fc = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activate3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activate4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activate5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN_Choi_Slim(Module):
    """
    """

    def __init__(self):
        super(CNN_Choi_Slim, self).__init__()

        # input block
        self.bn = nn.BatchNorm2d(96)

        # block1
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.activate1 = nn.ELU()
        self.maxpool1 = nn.MaxPool2d((2, 4))

        # block2
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.activate2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d((2, 4))

        # block3
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.activate3 = nn.ELU()
        self.maxpool3 = nn.MaxPool2d((2, 4))

        # block4
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.activate4 = nn.ELU()
        self.maxpool4 = nn.MaxPool2d((3, 5))

        # block5
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.activate5 = nn.ELU()
        self.maxpool5 = nn.MaxPool2d((4, 4))
        # self.hidden = nn.Linear(64, 256)

    def forward(self, x):
        # x = x.permute(0, 2, 1, 3)
        # x = self.bn(x)
        # x = x.permute(0, 2, 1, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activate3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activate4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activate5(x)
        x = self.maxpool5(x)

        x = x.view(x.size(0), -1)
        # x = self.hidden(x)

        # x = self.fc(x)
        return x


class RCNN_Choi(Module):
    def __init__(self):
        super(RCNN_Choi, self).__init__()

        # input block
        self.bn = nn.BatchNorm2d(96)

        # block1
        self.conv1 = nn.Conv2d(1, 64, 3, padding=(1, 38))

        self.bn1 = nn.BatchNorm2d(64)
        self.activate1 = nn.ELU()
        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.1)

        # block2
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.activate2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d((3, 3))
        self.dropout2 = nn.Dropout(0.1)

        # block3
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.activate3 = nn.ELU()
        self.maxpool3 = nn.MaxPool2d((4, 4))
        self.dropout3 = nn.Dropout(0.1)

        # block4
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.activate4 = nn.ELU()
        self.maxpool4 = nn.MaxPool2d((4, 4))
        self.dropout4 = nn.Dropout(0.1)

        self.gru1 = nn.GRU(128, 32)
        self.gru2 = nn.GRU(32, 32)
        # self.dropout5 = nn.Dropout(0.3)

        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        # x = x.permute(0, 2, 1, 3)
        # x = self.bn(x)
        # x = x.permute(0, 2, 1, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activate3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activate4(x)
        x = self.maxpool4(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(2, 0, 1)

        output1, hidden1 = self.gru1(x)
        output2, hidden2 = self.gru2(output1)
        x = hidden2
        # x = self.dropout5(x)
        x = x.squeeze(dim=0)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = CNN_Choi_Slim()

    # model = resnet34(pretrained=True)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model = nn.Sequential(*list(model.children())[:-1])
    audio_path = "../audio/bensound-actionable.mp3"
    mel_spectrum = compute_melgram(audio_path)
    mel_spectrum = torch.from_numpy(mel_spectrum).float()

    output = model(mel_spectrum).reshape(1,-1)
    print(output.shape)
