from util import audio_dataset
from torch.utils.data import DataLoader
from torchvision import models
import torch
from torch import nn
from torch.optim import Adam
from tensorboardX import SummaryWriter
from datetime import datetime
from model.cnn_baseline import CNN_Baseline
import os
import time
import copy

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, writer, num_epochs=100):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        info = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                # inputs, labels = inputs.cuda(device_ids[0]), labels.cuda(device_ids[0])

                labels = labels.squeeze()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # _, preds = outputs.topk(1, 1, True, True)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                sum = torch.sum(preds == labels.data)
                running_corrects += sum
            print("corrects sum {}".format(str(running_corrects)))
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            # epoch_acc = np.mean(mAP)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            info[phase] = {'acc': epoch_acc, 'loss': epoch_loss}

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        writer.add_scalars('data/acc', {'train': info["train"]['acc'], 'val': info["val"]['acc']}, epoch)
        writer.add_scalars('data/loss', {'train': info["val"]['loss'], 'val': info["val"]['loss']}, epoch)

        print()
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "music_cnn.pth")

    return model, val_acc_history


class Config():
    train_batch_size = 16
    val_batch_size = 16
    num_classes = 10


if __name__ == '__main__':
    writer = SummaryWriter(logdir=os.path.join("../tb_log", "font_" + datetime.now().strftime('%b%d_%H-%M-%S')))

    gtzan_datasets = audio_dataset.get_gtzan_datasets()
    gtzan_dataloaders = {x: DataLoader(gtzan_datasets[x], batch_size=16, shuffle=True, num_workers=8) for x in
                         ['train', 'val']}

    # model = models.resnet50(pretrained=True)
    model = CNN_Baseline()

    model = model.to(device)
    in_features = model.fc.in_features
    out_features = Config.num_classes
    model.fc = torch.nn.Linear(in_features, out_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_model(model, gtzan_dataloaders, criterion, optimizer, writer=writer)
