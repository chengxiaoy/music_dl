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

from glob import glob
from sklearn.model_selection import train_test_split


class GTZANDataSet(Dataset):
    def __init__(self, audio_infos):
        self.audio_infos = audio_infos

    def __getitem__(self, index):
        audio_info = self.audio_infos[index]
        mel_spectrum = compute_melgram(audio_info[0])
        return torch.Tensor(mel_spectrum).float(), torch.LongTensor([audio_info[1]])

    def __len__(self):
        return len(self.audio_infos)


def get_gtzan_datasets():
    genres_path = "../genres/"

    audio_paths = glob(genres_path + "*/*.au")

    genres_list = list(set([x.split('/')[-2] for x in audio_paths]))

    audio_infos = [(x, genres_list.index(x.split('/')[-2])) for x in audio_paths]

    train_audio_infos, test_audio_infos = train_test_split(audio_infos)

    audio_datasets = {'train': GTZANDataSet(train_audio_infos), 'val': GTZANDataSet(test_audio_infos)}
    return audio_datasets
