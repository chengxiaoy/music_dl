import torch
from torch import nn
import time
import _thread
import threading

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ThreadSafetyModel(nn.Module):

    def __init__(self):
        super(ThreadSafetyModel, self).__init__()
        self.sleep_time = 10

    def forward(self, *input):
        print(" in ==={}====".format(time.time()))
        print(input)
        time.sleep(self.sleep_time)
        print("=" * 30)
        print(" out ==={}====".format(time.time()))
        pass


def run_model(model, input):
    model(input)


model = ThreadSafetyModel()
model.to(device)

th1 = _thread.start_new_thread(run_model, (model, [1, 2, 3]))
th2 = _thread.start_new_thread(run_model, (model, [1, 2, 3]))

time.sleep(11)
print(th1)

## pytorch的forward 好像没有锁
