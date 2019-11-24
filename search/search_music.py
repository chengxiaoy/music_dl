import faiss
import numpy as np
import joblib
from model.siamese_model import *
from util.audio_processor import compute_melgram
import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

music_path = '../audio/data27075/541511427.mp3'
# music_path1 = '../util/149791.mp3'
# music_path2 = '../util/392907.mp3'

model = SiameseModel()
model.load_state_dict(torch.load("../data/music_siamese.pth", map_location='cpu'))
model.to(device)
model.eval()


def get_index(feature):
    d = len(feature[0])
    index = faiss.IndexFlatL2(d)  # the other index
    index.add(feature)
    return index


recall_num = 4000

features, paths = joblib.load('vec.pkl')
features = np.array(features)

index = get_index(features)
# feature1 = model.forward_once(torch.Tensor(compute_melgram(music_path1)).float()).detach().numpy()
# feature2 = model.forward_once(torch.Tensor(compute_melgram(music_path2)).float()).detach().numpy()
feature = model.forward_once(torch.Tensor(compute_melgram(music_path)).float()).detach().numpy()
D, I = index.search(np.array(feature, dtype=np.float32), recall_num)
#
# for i, (s_d, s_i) in enumerate(zip(D, I)):
#     print(s_d[1:])
#     for k in range(1, 4):
#         # if s_d[k] > 0.002:
#         #     break
#         print("search for {}<===>{}".format(paths[i], paths[s_i[k]]))

print(D[0][1000:])
print(paths[I[0][0]])
print(paths[I[0][1]])
print(paths[I[0][2]])
print(paths[I[0][3]])