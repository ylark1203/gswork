import numpy as np
from scipy.spatial import cKDTree
import igl

def barycentric_coords(P, A, B, C, eps=1e-12):
    """
    计算点 P 相对三角形 ABC 的重心坐标 (w0,w1,w2)，保证和为1。
    P,A,B,C: (3,) np.float64
    返回: (3,) np.float64
    """
    v0 = B - A
    v1 = C - A
    v2 = P - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        # 退化：退回到最近点归一平均
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)

def transfer_weights_5023_to_5083(
    V_5023, F_5023, V_5083, W_5023, 
    teeth_idx=None, method="closest_face"
):
    """
    将 5023 网格上的权重转移到 5083（含牙齿）：
    - 前 5023 个顶点直接拷贝
    - 牙齿 60 个顶点通过 'method' 从 5023 表面插值得到
    
    参数:
      V_5023: (5023, 3) float64
      F_5023: (F, 3)    int64
      V_5083: (5083, 3) float64
      W_5023: (5023, K) float64
      teeth_idx: (60,)  int64，默认 np.arange(5023,5083)
      method: "nn" 或 "closest_face"（推荐）
    
    返回:
      W_5083: (5083, K) float64
    """
    V_5023 = np.asarray(V_5023, dtype=np.float64, order='C')
    F_5023 = np.asarray(F_5023, dtype=np.int64, order='C')
    V_5083 = np.asarray(V_5083, dtype=np.float64, order='C')
    W_5023 = np.asarray(W_5023, dtype=np.float64, order='C')

    N_5023 = V_5023.shape[0]
    N_5083 = V_5083.shape[0]
    assert N_5023 == 5023, "期望 V_5023 顶点数为 5023"
    assert N_5083 == 5083, "期望 V_5083 顶点数为 5083"
    assert W_5023.shape[0] == 5023

    K = W_5023.shape[1]
    if teeth_idx is None:
        teeth_idx = np.arange(5023, 5083, dtype=np.int64)

    # 1) 初始化并直接拷贝前 5023
    W_5083 = np.zeros((N_5083, K), dtype=np.float64)
    W_5083[:5023] = W_5023

    # 2) 对牙齿 60 顶点做插值
    V_teeth = V_5083[teeth_idx]  # (60,3)

    if method == "nn":
        # --- 方法A：最近邻 ---
        tree = cKDTree(V_5023)
        _, nn_idx = tree.query(V_teeth, k=1)
        W_5083[teeth_idx] = W_5023[nn_idx]
    else:
        # --- 方法B：表面最近三角形 + 重心插值（推荐） ---
        # 获取每个牙齿点在 5023 表面上的最近点 C、对应三角形索引 I
        # sqrD: (T,), I: (T,), C: (T,3)
        sqrD, I, C = igl.point_mesh_squared_distance(V_teeth, V_5023, F_5023)
        # 用重心坐标把三角形顶点的权重插值到 C
        for t in range(V_teeth.shape[0]):
            tri = F_5023[I[t]]                      # (3,)
            A, B, Ctri = V_5023[tri[0]], V_5023[tri[1]], V_5023[tri[2]]
            P = C[t]                                # 最近点坐标（在三角形上）
            bc = barycentric_coords(P, A, B, Ctri)  # (3,)
            # 以重心系数加权三顶点的权重
            W_tri = W_5023[tri]                     # (3, K)
            W_5083[teeth_idx[t]] = bc @ W_tri       # (K,)

    # 3) 清理与归一
    W_5083 = np.nan_to_num(W_5083, nan=0.0)
    W_5083 = np.clip(W_5083, 0.0, 1.0)
    row_sum = W_5083.sum(axis=1, keepdims=True) + 1e-12
    W_5083 /= row_sum
    return W_5083


template = np.load("/mnt/data/lyl/codes/RGBAvatar/BBW/vertice_and_faces/5083vertices.npz")
V_5083 = template.f.v_template
F_5083 = template.f.faces
V_5023 = V_5083[:5023]
F_5023 = F_5083[:9976]
W_5023 = np.load("/mnt/data/lyl/codes/RGBAvatar/BBW/vertice_and_faces_500/5023_500.npz")
handle_indices = W_5023.f.handle_indices
W_5023 = W_5023.f.vertex_bbw

# 已有的：
# V_5023, F_5023 from FLAME 原模
# W_5023: 你在 5023 网格上已经算好的 [5023, K] 权重
# V_5083, F_5083: 含牙齿的拓扑（add_teeth 后）

# 方式1：更稳（推荐）
W_5083 = transfer_weights_5023_to_5083(V_5023, F_5023, V_5083, W_5023, method="closest_face")
np.savez("/mnt/data/lyl/codes/RGBAvatar/BBW/vertice_and_faces_500/5083_500_bbw.npz", vertex_bbw=W_5083, handle_indices=handle_indices)
# 方式2：更快（当做 baseline/应急）
# W_5083 = transfer_weights_5023_to_5083(V_5023, F_5023, V_5083, W_5023, method="nn")
