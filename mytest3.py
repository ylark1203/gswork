import torch
import torch.nn as nn
import numpy as np
import pickle as pkl


"""
    J.shape: [5, 3]
    weights.shape: [5023, 5]
    posedirs.shape: [5023, 3, 36]
    v_template.shape: [5023, 3]
    shapedirs.shape: [5023, 3, 400]
    J_regressor.shape: [5, 5023]
"""

a = pkl.load(open("/mnt/data/lyl/codes/RGBAvatar/data/FLAME2020/generic_model.pkl", 'rb'), encoding='latin1')
b = torch.load("/mnt/data/lyl/datasets/INSTA/marcel/checkpoint/00000.frame", weights_only=False)
c = np.load("/mnt/data/lyl/codes/RGBAvatar/data/FLAME2020/flame_uv.npz")
# c = np.load("/mnt/data/lyl/codes/RGBAvatar/BBW/20.npz")
# d = np.load("/mnt/data/lyl/codes/RGBAvatar/BBW/vertice_and_faces/5083vertices.npz")
1