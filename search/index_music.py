from util import audio_processor
from model.siamese_model import SiameseModel
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import joblib
import os
from glob import glob
from util.audio_dataset import get_mel, split_n_melgram
from torch import nn
from tqdm import tqdm
from train_music163 import *

warnings.filterwarnings('ignore')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class MusicDataset(Dataset):

    def __init__(self, music_paths, pair=True, multi=False):
        super(MusicDataset, self).__init__()
        self.music_paths = music_paths
        self.multi = multi
        self.pair = pair

    def __len__(self):
        return len(self.music_paths)

    def __getitem__(self, index):
        try:
            if self.multi:
                return audio_processor.compute_melgram_multi_slice(self.music_paths[index]), self.music_paths[index]
            else:

                if self.pair:
                    return torch.Tensor(split_n_melgram(get_mel(self.music_paths[index])[0])[0]).float(), \
                           self.music_paths[index]
                else:
                    return torch.Tensor(get_mel(self.music_paths[index])[0]).float(), self.music_paths[index]
        except Exception as e:
            print(e)
            return None, self.music_paths[index]


def collate_double(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


def getFinetuneModel(config, weight_path):
    config.train_batch_size = 1
    config.val_batch_size = 1
    model = get_model(config)
    model.load_state_dict(torch.load(weight_path))
    if config.multi_gpu:
        model = model.module
    model.to(device)
    model.eval()

    return model


def full_index_v1(paths, config, model=None):
    # load multi gpu weights

    if model == None:
        model = getFinetuneModel(config, "music_siamese_50000Nov28_07-11-11.pth")

    m_dataset = MusicDataset(paths, config.dataset_pair, False)
    data_loader = DataLoader(m_dataset, shuffle=False, num_workers=8, batch_size=64, collate_fn=collate_double,
                             drop_last=True)
    vec_list = []
    path_list = []

    for mels, paths in tqdm(data_loader):
        for mel, path in zip(mels, paths):
            if mel is None:
                continue
            if config.model_type == "crnn":
                output = model.forward_once((mel.to(device), model.init_h0().to(device)))
            else:
                output = model.forward_once(mel.to(device))
            vec = output[0].cpu().detach().numpy()
            vec_list.append(vec)
            path_list.append(path)
            print("extract vect from {}".format(path))
    joblib.dump((vec_list, path_list), "vec_28_07-11-11.pkl")


if __name__ == '__main__':
    music_path = "./audio/"
    audio_paths = glob(music_path + "*/*.mp3")
    print(len(audio_paths))
    config = Config()
    full_index_v1(audio_paths, config, None)
