import torch
from torch import nn
from model.cnn_choi import CNN_Choi_Slim
from util import audio_processor
from torch.nn import functional


class SiameseModel(nn.Module):

    def __init__(self):
        super(SiameseModel, self).__init__()
        self.backbone = CNN_Choi_Slim()
        self.ll = nn.Linear(512, 100)
        self.relu = nn.ReLU()
        self.ll2 = nn.Linear(100, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, input1, input2):
        output1 = self.backbone(input1)
        output2 = self.backbone(input2)

        # (x1-x2)**2
        sub = torch.sub(output1, output2)
        mul1 = torch.mul(sub, sub)

        # (x1**2-x2**2)
        mul2 = torch.sub(torch.mul(output1, output1), torch.mul(output2, output2))

        x = torch.cat([mul1, mul2], 1)
        x = x.view(x.size(0), -1)

        x = self.ll(x)
        x = self.relu(x)

        x = self.ll2(x)
        x = self.sigmod(x)
        return x


if __name__ == '__main__':
    model = SiameseModel()

    music_clip = audio_processor.compute_melgram("../audio/Fabel-不生不死.mp3")

    CNN_Choi_Slim()(torch.Tensor(music_clip).float())

    x = model(torch.Tensor(music_clip).float(), torch.Tensor(music_clip).float())
    print(x)
