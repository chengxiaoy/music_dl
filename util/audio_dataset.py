from glob import glob
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from random import choice, sample
import numpy as np
import PIL
from PIL import Image
import torchvision
import torch
from util.audio_processor import *
import h5py
import os
from glob import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib
from util.ecoding import parse_md5
from multiprocessing import Pool

audio_mel_dir = "audio/"


class GTZANDataSet(Dataset):
    def __init__(self, audio_infos, version=1):
        self.audio_infos = audio_infos
        self.version = version

    def __getitem__(self, index):
        audio_info = self.audio_infos[index]

        audio_path = audio_info[0]
        if self.version == 1:
            md5 = parse_md5(audio_path)
            mel_path = audio_mel_dir + md5 + ".pkl"
        elif self.version == 2:
            md5 = parse_md5(audio_path + "version:2")
            mel_path = audio_mel_dir + md5 + ".pkl"
        if not os.path.exists(mel_path):
            audio2mel(audio_path, mel_path)

        mel_spectrum = joblib.load(mel_path)

        return torch.Tensor(mel_spectrum[0]).float(), torch.LongTensor([audio_info[1]])

    def __len__(self):
        return len(self.audio_infos)


def audio2mel(audio_path, mel_path, version=1):
    if version == 1:
        mel_spectrum = compute_melgram(audio_path)
    elif version == 2:
        mel_spectrum = compute_melgram(audio_path, SR=22050, N_FFT=2048, N_MELS=128, HOP_LEN=1024, DURA=30)
    joblib.dump(mel_spectrum, mel_path)


def audio2Multimel(paths):
    try:
        multi_mel_spectrum = compute_melgram_multi_slice(paths[0])
        joblib.dump(multi_mel_spectrum, paths[1])
    except Exception as e:
        print(e)


def get_gtzan_datasets(version=1):
    genres_path = "../genres/"
    audio_paths = glob(genres_path + "*/*.au")

    genres_list = list(set([x.split('/')[-2] for x in audio_paths]))

    audio_infos = [(x, genres_list.index(x.split('/')[-2])) for x in audio_paths]
    audio_labels = [genres_list.index(x.split('/')[-2]) for x in audio_paths]

    # StratifiedKFold
    # sfolder = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    # train_index, test_index = list(sfolder.split(audio_infos, audio_labels))[0]
    #
    # train_audio_infos = []
    # test_audio_infos = []
    # for index in train_index:
    #     train_audio_infos.append(audio_infos[index])
    # for index in test_index:
    #     test_audio_infos.append(audio_infos[index])

    train_audio_infos, test_audio_infos = train_test_split(audio_infos)

    audio_datasets = {'train': GTZANDataSet(train_audio_infos, version), 'val': GTZANDataSet(test_audio_infos, version)}
    return audio_datasets


def get_mel(audio_path):
    mel_path = audio_path.replace(".mp3", ".pkl")
    if os.path.exists(mel_path):
        melgram_list = joblib.load(mel_path)
    else:
        melgram_list = compute_melgram_multi_slice(audio_path)
    return melgram_list


class SiameseDataSet(Dataset):
    def __init__(self, audio_infos):
        self.audio_infos = audio_infos

    def __getitem__(self, index):
        while True:
            try:
                if index % 2 == 0:
                    audio_path, audio_label = self.audio_infos[(index // 2) % len(self.audio_infos)]
                    melgram1, melgram2 = sample(get_mel(audio_path), 2)
                    # print("index {} melgram===>{}".format(index, audio_path))
                    return torch.Tensor(melgram1[0]).float(), torch.Tensor(melgram2[0]).float(), torch.Tensor([1])
                else:
                    audio_info1, audio_info2 = sample(self.audio_infos, 2)
                    audio_path1, audio_path2 = audio_info1[0], audio_info2[0]
                    melgram1, melgram2 = choice(get_mel(audio_path1)), choice(get_mel(audio_path2))
                    # print("index {} melgram===>{},{}".format(index, audio_path1, audio_path2))
                    return torch.Tensor(melgram1[0]).float(), torch.Tensor(melgram2[0]).float(), torch.Tensor([0])
            except Exception as e:
                # print("get item {} error {}".format(index, e))
                # 读取音频可能报错
                return self.__getitem__(index + 2)

    def __len__(self):
        return len(self.audio_infos) * 2


class SiameseNeighborDataSet(Dataset):
    def __init__(self, audio_infos):
        self.audio_infos = audio_infos

    def __getitem__(self, index):
        while True:
            try:
                if index % 2 == 0:
                    audio_path, audio_label = self.audio_infos[(index // 2) % len(self.audio_infos)]
                    melgram = choice(get_mel(audio_path))
                    melgram1, melgram2 = split_n_melgram(melgram)
                    # print("index {} melgram===>{}".format(index, audio_path))
                    return torch.Tensor(melgram1[0]).float(), torch.Tensor(melgram2[0]).float(), torch.Tensor([1])
                else:
                    audio_info1, audio_info2 = sample(self.audio_infos, 2)
                    audio_path1, audio_path2 = audio_info1[0], audio_info2[0]
                    melgram1_, melgram2_ = choice(get_mel(audio_path1)), choice(get_mel(audio_path2))
                    melgram1, _ = split_n_melgram(melgram1_)
                    melgram2, _ = split_n_melgram(melgram2_)
                    # print("index {} melgram===>{},{}".format(index, audio_path1, audio_path2))
                    return torch.Tensor(melgram1[0]).float(), torch.Tensor(melgram2[0]).float(), torch.Tensor([0])
            except Exception as e:
                # print("get item {} error {}".format(index, e))
                # 读取音频可能报错
                return self.__getitem__(index + 2)

    def __len__(self):
        return len(self.audio_infos) * 2


def split_n_melgram(melgram):
    return melgram[:, :, :, :480], melgram[:, :, :, 886:]


def get_siamese_datasets(size, pair=True):
    genres_path = "./audio/"
    audio_paths = glob(genres_path + "*/*.mp3")
    audio_paths = sorted(audio_paths)
    audio_paths = audio_paths[:size]
    genres_list = list(set([x.split('/')[-2] for x in audio_paths]))

    audio_infos = [(x, genres_list.index(x.split('/')[-2])) for x in audio_paths]

    print(len(audio_infos))

    # StratifiedKFold
    # sfolder = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    # train_index, test_index = list(sfolder.split(audio_infos, audio_labels))[0]
    #
    # train_audio_infos = []
    # test_audio_infos = []
    # for index in train_index:
    #     train_audio_infos.append(audio_infos[index])
    # for index in test_index:
    #     test_audio_infos.append(audio_infos[index])

    train_audio_infos, test_audio_infos = train_test_split(audio_infos)

    if pair:
        audio_datasets = {'train': SiameseNeighborDataSet(train_audio_infos),
                          'val': SiameseNeighborDataSet(test_audio_infos)}
    else:
        audio_datasets = {'train': SiameseDataSet(train_audio_infos), 'val': SiameseDataSet(test_audio_infos)}
    return audio_datasets


if __name__ == '__main__':
    genres_path = "./audio/"
    audio_paths = glob(genres_path + "data*/*.mp3")
    mel_paths = [x.replace('.mp3', '.pkl') for x in audio_paths]
    print(mel_paths)
    p = Pool(16)
    args = []
    for audio_path, mel_path in zip(audio_paths, mel_paths):
        args.append((audio_path, mel_path))

    p.map(audio2Multimel, args)
