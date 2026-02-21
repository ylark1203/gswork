import torch
import pickle as pkl

a = torch.load("/mnt/data/lyl/datasets/INSTA/bala/checkpoint/01990.frame", weights_only=False)
b = pkl.load(open("/mnt/data/lyl/codes/RGBAvatar/data/FLAME2020/generic_model.pkl", 'rb'), encoding='latin1')
1