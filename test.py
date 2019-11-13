import torch
from torch import nn
import time
import _thread
import threading
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ThreadSafetyModel(nn.Module):

    def __init__(self):
        super(ThreadSafetyModel, self).__init__()
        self.sleep_time = 20

    def forward(self, input):
        print(" in ==={}====".format(time.time()))
        memory_1G_gpu = torch.Tensor(input).float().half().to(device)
        time.sleep(self.sleep_time)
        print("=" * 30)
        print(" out ==={}====".format(time.time()))
        pass


def run_model(model, input):
    model(input)


model = ThreadSafetyModel()
model.to(device)
memory_1G = np.random.rand(1024, 1024, 256)

th1 = _thread.start_new_thread(run_model, (model, memory_1G))
th2 = _thread.start_new_thread(run_model, (model, memory_1G))

time.sleep(31)
print(th1)
print(th2)

## pytorch的forward 好像没有锁
