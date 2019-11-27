import torch
from torch import nn
from model.cnn_choi import CNN_Choi_Slim, CNN_Choi
from util import audio_processor
from torch.nn import functional
from torchvision.models import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SiameseModel(nn.Module):

    def __init__(self):
        super(SiameseModel, self).__init__()
        model = resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model = nn.Sequential(*list(model.children())[:-1])
        self.backbone = model
        self.normal = nn.functional.normalize

        # self.backbone = CNN_Choi()

        self.ll = nn.Linear(1024, 100)
        self.relu = nn.ReLU()
        self.ll2 = nn.Linear(100, 1)
        self.sigmod = nn.Sigmoid()

    def forward_once(self, input):
        output = self.backbone(input)
        output = output.squeeze(-1).squeeze(-1)
        output = self.normal(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class SiameseModelRNN(nn.Module):
    def __init__(self):
        super(SiameseModelRNN, self).__init__()
        model = resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model = nn.Sequential(*list(model.children())[:-2])
        self.gru = nn.GRU(512, 512, batch_first=True, bidirectional=True)
        self.backbone = model
        self.normal = nn.functional.normalize
        self.pool = nn.MaxPool2d((3, 1))

    def forward_once_rnn(self, input):
        cnn_input, h0 = input
        output = self.backbone(cnn_input)
        output = self.pool(output).squeeze(dim=2)
        rnn_in = output.permute([0, 2, 1])
        output, hn = self.gru(rnn_in, h0)
        b = hn.shape[1]
        hn = hn.reshape(b, -1)
        return self.normal(hn)

    def forward(self, input1, input2):
        output1 = self.forward_once_rnn(input1)
        output2 = self.forward_once_rnn(input2)

        return output1, output2


if __name__ == '__main__':
    model = SiameseModelRNN()

    music_clip = audio_processor.compute_melgram("../audio/Fabel-不生不死.mp3")
    x = model((torch.Tensor(music_clip).float(), torch.zeros(2, 1, 512)),
              (torch.Tensor(music_clip).float(), torch.zeros(2, 1, 512)))
    print(x)
