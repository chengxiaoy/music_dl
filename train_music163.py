from util.audio_dataset import *
from util import audio_dataset
from model import loss
import time
import copy
from tensorboardX import SummaryWriter
from datetime import datetime
from model.cnn_choi import *
from model.siamese_model import SiameseModel
from torch.optim import Adam
import torch.nn.functional as F

import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, writer, num_epochs=150):
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
            for input1s, input2s, labels in dataloaders[phase]:
                print("=======forward========")
                input1s = input1s.to(device)
                input2s = input2s.to(device)
                labels = labels.to(device)
                labels = labels.squeeze()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    output1s, output2s = model(input1s, input2s)
                    loss = criterion(output1s, output2s, labels)
                    # _, preds = outputs.topk(1, 1, True, True)
                    # _, preds = torch.max(outputs, 1)

                    threshold = 0.5
                    preds = F.pairwise_distance(output1s, output2s) > threshold

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        print("=======backward========")

                        loss.backward()
                        # nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                sum = torch.sum(preds.byte() == labels.byte())
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
        writer.add_scalars('data/loss', {'train': info["train"]['loss'], 'val': info["val"]['loss']}, epoch)

        print()
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "music_siamese.pth")

    return model, val_acc_history


class Config():
    train_batch_size = 16
    val_batch_size = 16


if __name__ == '__main__':
    writer = SummaryWriter(logdir=os.path.join("../tb_log", "163muisc_" + datetime.now().strftime('%b%d_%H-%M-%S')))

    siamese_datasets = audio_dataset.get_siamese_datasets()
    siamese_dataloaders = {
        x: DataLoader(siamese_datasets[x], batch_size=16, pin_memory=True, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    model = SiameseModel()
    model = model.to(device)

    criterion = loss.ContrastiveLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_model(model, siamese_dataloaders, criterion, optimizer, writer=writer)
