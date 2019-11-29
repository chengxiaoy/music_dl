from util.audio_dataset import *
from util import audio_dataset
from model import loss
import time
import copy
from tensorboardX import SummaryWriter
from datetime import datetime
from model.cnn_choi import *
from model.siamese_model import SiameseModel, SiameseModelRNN
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
warnings.filterwarnings('ignore')


def train_model(model, dataloaders, criterion, optimizer, writer, scheduler, config, num_epochs=150):
    since = time.time()
    val_acc_history = []
    saved_model_name = "music_siamese_50000" + datetime.now().strftime('%b%d_%H-%M-%S') + ".pth"

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    stop_times = 0

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

            sum_label = torch.zeros(0)
            sum_preds = torch.zeros(0)
            for input1s, input2s, labels in tqdm(dataloaders[phase]):

                if config.multi_gpu:
                    if config.model_type == 'cnn':
                        input1s = input1s.cuda(config.device_ids[0])
                        input2s = input2s.cuda(config.device_ids[0])
                        labels = labels.cuda(config.device_ids[0])
                    else:
                        input1s = input1s.cuda(config.device_ids[0]), model.init_h0().cuda(config.device_ids[0])
                        input2s = input2s.cuda(config.device_ids[0]), model.init_h0().cuda(config.device_ids[0])
                        labels = labels.cuda(config.device_ids[0])

                else:
                    if config.model_type == 'cnn':
                        input1s = input1s.to(config.device)
                        input2s = input2s.to(config.device)
                        labels = labels.to(config.device)
                    else:
                        input1s = input1s.to(config.device), model.init_h0().to(config.device)
                        input2s = input2s.to(config.device), model.init_h0().to(config.device)
                        labels = labels.to(config.device)
                labels = labels.squeeze()

                sum_label = torch.cat((sum_label, labels.cpu()))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    output1s, output2s = model(input1s, input2s)
                    loss = criterion(output1s, output2s, labels)
                    # _, preds = outputs.topk(1, 1, True, True)
                    # _, preds = torch.max(outputs, 1)

                    preds = F.pairwise_distance(output1s, output2s) < config.evalue_thr
                    sum_preds = torch.cat((sum_preds, preds.cpu().float()))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)
                        optimizer.step()

                # statistics
                running_loss += loss.item()
            running_corrects = torch.sum(sum_preds.byte() == sum_label.byte())
            tp, fp, tn, fn = 0, 0, 0, 0
            for pred, label in zip(sum_preds.byte(), sum_label.byte()):
                if pred == label:
                    if pred == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if pred == 1:
                        fp += 1
                    else:
                        fn += 1

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_tpr = tp * 2 / len(dataloaders[phase].dataset)
            epoch_tnr = tn * 2 / len(dataloaders[phase].dataset)
            print("corrects sum {}".format(str(running_corrects)))
            print("epoch_tpr: {}".format(str(epoch_tpr)))
            print("epoch_tnr: {}".format(str(epoch_tnr)))

            # epoch_acc = np.mean(mAP)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            info[phase] = {'acc': epoch_acc, 'loss': epoch_loss}

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                stop_times = 0
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), saved_model_name)
            if phase == 'val' and epoch_acc < best_acc:
                stop_times = stop_times + 1

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        writer.add_scalars('data/acc', {'train': info["train"]['acc'], 'val': info["val"]['acc']}, epoch)
        writer.add_scalars('data/loss', {'train': info["train"]['loss'], 'val': info["val"]['loss']}, epoch)
        scheduler.step(info["val"]['loss'])
        if stop_times >= 20:
            break
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), saved_model_name)

    return model, val_acc_history


class Config():
    train_batch_size = 64
    val_batch_size = 64
    model_type = 'crnn'  # cnn
    evalue_thr = 0.2
    dataset_size = 10000
    dataset_pair = True

    device_ids = [0, 1]
    backbone_type = 'resnet'  # choi
    multi_gpu = False
    single_gpu_id = 0
    device = torch.device("cuda:" + str(single_gpu_id) if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    config = Config()
    writer = SummaryWriter(logdir=os.path.join("../tb_log", "163muisc_" + datetime.now().strftime('%b%d_%H-%M-%S')))

    siamese_datasets = audio_dataset.get_siamese_datasets(config.dataset_size, pair=config.dataset_pair)
    siamese_dataloaders = {
        x: DataLoader(siamese_datasets[x], batch_size=config.train_batch_size, shuffle=True, num_workers=16,
                      drop_last=True) for x in
        ['train', 'val']}

    if config.model_type == 'crnn':
        model = SiameseModelRNN(batch_size=config.train_batch_size)
    elif config.model_type == 'cnn':
        model = SiameseModel(type=config.backbone_type)

    if config.multi_gpu:
        model = model.cuda(config.device_ids[0])
        model = nn.DataParallel(model, device_ids=config.device_ids)
    else:
        model = model.to(config.device)

    criterion = loss.ContrastiveLoss(margin=1.6)
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.1, verbose=True)
    train_model(model, siamese_dataloaders, criterion, optimizer, writer, scheduler, config, num_epochs=100)
