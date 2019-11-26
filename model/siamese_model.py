import torch
from torch import nn
from model.cnn_choi import CNN_Choi_Slim,CNN_Choi
from util import audio_processor
from torch.nn import functional
from torchvision.models import resnet34


class SiameseModel(nn.Module):

    def __init__(self):
        super(SiameseModel, self).__init__()
        model = resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.Sequential(*list(model.children())[:-1])
        # self.backbone = CNN_Choi()
        self.backbone = model

        self.ll = nn.Linear(1024, 100)
        self.relu = nn.ReLU()
        self.ll2 = nn.Linear(100, 1)
        self.normal = nn.functional.normalize
        self.sigmod = nn.Sigmoid()

    def forward_once(self, input):
        output = self.backbone(input)
        output = output.squeeze(-1).squeeze(-1)
        output = self.normal(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1,output2

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
    model = SiameseModel()

    music_clip = audio_processor.compute_melgram("../audio/Fabel-不生不死.mp3")
    x = model(torch.Tensor(music_clip).float(), torch.Tensor(music_clip).float())
    print(x)
