import torch
from torch import nn
from model.cnn_choi import CNN_Choi_Slim, CNN_Choi
from util import audio_processor
from torch.nn import functional
from torchvision.models import resnet34


class SiameseModel(nn.Module):

    def __init__(self, rnn=False):
        super(SiameseModel, self).__init__()
        model = resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.rnn = rnn
        if not rnn:
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            model = nn.Sequential(*list(model.children())[:-2])
        self.backbone = model
        self.normal = nn.functional.normalize

        self.gru = nn.GRU(512, 512, bidirectional=True)

        # self.backbone = CNN_Choi()

        self.pool = nn.MaxPool2d((3, 1))
        self.ll = nn.Linear(1024, 100)
        self.relu = nn.ReLU()
        self.ll2 = nn.Linear(100, 1)
        self.sigmod = nn.Sigmoid()

    def forward_once(self, input):
        output = self.backbone(input)
        if self.rnn:
            h0 = torch.zeros(2, 1, 512)
            output = self.pool(output).squeeze(dim=2)
            rnn_in = output.permute([2, 0, 1])
            output, hn = self.gru(rnn_in, h0)
            b = hn.shape[1]
            hn = hn.permute([1, 0, 2]).reshape(b, -1)
            return hn

        output = output.squeeze(-1).squeeze(-1)
        output = self.normal(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

        # (x1-x2)**2
        # sub = torch.sub(output1, output2)
        # mul1 = torch.mul(sub, sub)
        #
        # # (x1**2-x2**2)
        # mul2 = torch.sub(torch.mul(output1, output1), torch.mul(output2, output2))
        #
        # x = torch.cat([mul1, mul2], 1)
        # x = x.view(x.size(0), -1)
        #
        # x = self.ll(x)
        # x = self.relu(x)
        #
        # x = self.ll2(x)
        # x = self.sigmod(x)
        # return x


if __name__ == '__main__':
    model = SiameseModel(rnn=True)

    music_clip = audio_processor.compute_melgram("../audio/Fabel-不生不死.mp3")
    x = model(torch.Tensor(music_clip).float(), torch.Tensor(music_clip).float())
    print(x)
