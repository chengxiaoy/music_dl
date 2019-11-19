from util import audio_processor
from model.siamese_model import SiameseModel
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import joblib
import os
from glob import glob

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MusicDataset(Dataset):

    def __init__(self, music_paths, multi=False):
        super(MusicDataset, self).__init__()
        self.music_paths = music_paths
        self.multi = multi

    def __len__(self):
        return len(self.music_paths)

    def __getitem__(self, index):
        try:
            if self.multi:
                return audio_processor.compute_melgram_multi_slice(self.music_paths[index]), self.music_paths[index]
            else:
                return audio_processor.compute_melgram(self.music_paths[index]), self.music_paths[index]
        except Exception as e:
            print(e)
            return None, self.music_paths[index]


def collate_double(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


def full_index_v1(paths):
    model = SiameseModel()
    model.load_state_dict(torch.load("music_siamese.pth"))
    model.to(device)
    model.eval()

    m_dataset = MusicDataset(paths, False)
    data_loader = DataLoader(m_dataset, shuffle=False, num_workers=4, batch_size=8, collate_fn=collate_double)
    vec_list = []
    path_list = []

    for mels, paths in data_loader:
        for mel, path in zip(mels, paths):
            if mel == None:
                continue
            output = model.forward_once(mel.to(device))
            vec = output[0].cpu().numpy()
            vec_list.append(vec)
            path_list.append(path)
    joblib.dump((vec_list, path_list), "vec.pkl")


if __name__ == '__main__':
    music_path = "../audio/"
    audio_paths = glob(music_path + "*/*.mp3")
    full_index_v1(audio_paths)
