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


class SiameseDataSet(Dataset):
    def __init__(self, audio_infos):
        self.audio_infos = audio_infos

    def __getitem__(self, index):
        while True:
            try:
                if index % 2 == 0:
                    audio_path, audio_label = self.audio_infos[index // 2]

                    melgram_list = compute_melgram_multi_slice(audio_path)
                    melgram1, melgram2 = sample(melgram_list, 2)

                    print("index {} melgram===>{}".format(index, audio_path))
                    return torch.Tensor(melgram1[0]).float(), torch.Tensor(melgram2[0]).float(), torch.Tensor([1])
                else:
                    audio_info1, audio_info2 = sample(self.audio_infos, 2)
                    audio_path1, audio_path2 = audio_info1[0], audio_info2[0]

                    melgram1, melgram2 = choice(compute_melgram_multi_slice(audio_path1)), choice(
                        compute_melgram_multi_slice(audio_path2))

                    print("index {} melgram===>{},{}".format(index, audio_path1, audio_path2))
                    return torch.Tensor(melgram1[0]).float(), torch.Tensor(melgram2[0]).float(), torch.Tensor([0])
            except Exception as e:
                # 读取音频可能报错
                return self.__getitem__(index + 2)

    def __len__(self):
        return len(self.audio_infos) * 2


def get_siamese_datasets():
    genres_path = "./audio/"
    audio_paths = glob(genres_path + "*/*.mp3")

    genres_list = list(set([x.split('/')[-2] for x in audio_paths]))

    audio_infos = [(x, genres_list.index(x.split('/')[-2])) for x in audio_paths]

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

    audio_datasets = {'train': SiameseDataSet(train_audio_infos), 'val': SiameseDataSet(test_audio_infos)}
    return audio_datasets
