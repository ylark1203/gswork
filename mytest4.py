import numpy as np
import trimesh

data = np.load("/mnt/data/lyl/codes/RGBAvatar/BBW/vertice_and_faces/5083vertices.npz")
V = data.f.v_template   # 或 data["verts"]
F = data.f.faces     # 或 data["triangles"]

mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
mesh.export("/mnt/data/lyl/codes/RGBAvatar/model/transform/flame5083.ply")  # 默认会输出较标准的 ply
print("exported mesh.ply")

