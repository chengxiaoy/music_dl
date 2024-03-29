import faiss
import numpy as np
import joblib
from model.siamese_model import *
from util.audio_processor import compute_melgram, compute_melgram_multi_slice
import torch
from sklearn.neighbors import BallTree
from util.download_music163 import dowload_song
from sklearn.preprocessing import normalize

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

song_id = 29966565

path = '../music_dl/audio/data' + str(song_id // 20000) + "/" + str(song_id) + '.mp3'
music_path = '../audio/data' + str(song_id // 20000) + "/" + str(song_id) + '.mp3'


def get_index(feature):
    d = len(feature[0])
    index = faiss.IndexFlatL2(d)  # the other index
    index.add(feature)
    return index


recall_num = 10

paths, features = joblib.load('features.pkl')
# features, paths = joblib.load('vec_28_07-11-11.pkl')

# with open("path.txt") as f:
#     paths = f.readlines()
paths = [x.strip('\n') for x in paths]
features = normalize(features)
print(len(features))
# print(features[0])
# print(paths[0])

features = np.array(features).astype(np.float32)
index = get_index(features)

feature = np.array([features[paths.index(path)]])

D, I = index.search(feature, recall_num)

for d, i in zip(D[0], I[0]):
    print(d)
    print(paths[i])
