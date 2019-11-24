import torch
from torch import nn
import time
import _thread
import threading
import numpy as np

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class ThreadSafetyModel(nn.Module):

    def __init__(self):
        super(ThreadSafetyModel, self).__init__()
        self.sleep_time = 20

    def forward(self, input):
        print(" in ==={}====".format(time.time()))
        # memory_1G_gpu = torch.Tensor(input).float().to(device)

        memory_1G = np.random.rand(1024*1024, 768)
        memory_1G_ = np.random.rand(768, 1024*1024)
        print("===begin put gpu=====")


        memory_1G_gpu = torch.Tensor(memory_1G).float().to(device)
        memory_1G_gpu_ = torch.Tensor(memory_1G_).float().to(device)
        print("===begin computer=====")

        res = memory_1G_gpu_.mm(memory_1G_gpu)
        print(res.shape)
        print(memory_1G.shape)
        print(memory_1G_.shape)

        # time.sleep(self.sleep_time)
        print("=" * 30)
        print(" out ==={}====".format(time.time()))
        pass


def run_model(model, input):
    model(input)


model = ThreadSafetyModel()
model.to(device)

arg = 'hehe'

th1 = _thread.start_new_thread(run_model, (model, arg))
th2 = _thread.start_new_thread(run_model, (model, arg))

time.sleep(31)
print(th1)
print(th2)

## pytorch的forward 好像没有锁
