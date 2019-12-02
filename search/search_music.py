import faiss
import numpy as np
import joblib
from model.siamese_model import *
from util.audio_processor import compute_melgram, compute_melgram_multi_slice
import torch
from sklearn.neighbors import BallTree
from util.download_music163 import dowload_song

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

song_id = 28018075

path = './audio/data' + str(song_id // 20000) + "/" + str(song_id) + '.mp3'
music_path = '../audio/data' + str(song_id // 20000) + "/" + str(song_id) + '.mp3'

song_id1 = 308299
path1 = './audio/data' + str(song_id1 // 20000) + "/" + str(song_id1) + '.mp3'

music_path1 = '../audio/data' + str(song_id1 // 20000) + "/" + str(song_id1) + '.mp3'


# music_path = dowload_song(song_id)
# music_path1 = '../util/149791.mp3'
# music_path2 = '../util/392907.mp3'

# model = SiameseModel()
# model = nn.DataParallel(model)
# model.load_state_dict(torch.load("music_siamese_50000Nov27_02-38-26.pth", map_location='cpu'))
# model = model.module
# model.to(device)
# model.eval()


def get_index(feature):
    d = len(feature[0])
    index = faiss.IndexFlatL2(d)  # the other index
    index.add(feature)
    return index


recall_num = 10

features, paths = joblib.load('vec_28_07-11-11.pkl')
# features, paths = joblib.load('vec.pkl')
features = np.array(features)
index = get_index(features)
# tree = BallTree(features)

# feature1 = model.forward_once(torch.Tensor(compute_melgram(music_path1)).float()).detach().numpy()
# feature2 = model.forward_once(torch.Tensor(compute_melgram(music_path2)).float()).detach().numpy()
# feature_ = model.forward_once(torch.Tensor(compute_melgram_multi_slice(music_path)[0]).float()).detach().numpy()
feature = np.array([features[paths.index(path)]])
feature1 = np.array([features[paths.index(path1)]])
print(feature.dot(feature1.T))
# print(feature[0][:20])
# print(feature1[0][:20])

D, I = index.search(feature, recall_num)
# D, I = tree.query(feature, 4)
# for i, (s_d, s_i) in enumerate(zip(D, I)):
#     print(s_d[1:])
#     for k in range(1, 4):
#         # if s_d[k] > 0.002:
#         #     break
#         print("search for {}<===>{}".format(paths[i], paths[s_i[k]]))

print(D[0])
print(paths[I[0][0]])
print(paths[I[0][1]])
print(paths[I[0][2]])
print(paths[I[0][3]])
print(paths[I[0][4]])
print(paths[I[0][5]])
print(paths[I[0][6]])
print(paths[I[0][7]])
print(paths[I[0][8]])
print(paths[I[0][9]])
