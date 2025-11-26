import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
import igl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib import cm


# ========= 1. 读取 FLAME 模板 =========
template_path = "/mnt/data/lyl/codes/RGBAvatar/data/FLAME2020/generic_model.pkl"
template = pkl.load(open(template_path, 'rb'), encoding='latin1')

V = np.asarray(template['v_template'], dtype=np.float64)  # [V,3]
F = np.asarray(template['f'], dtype=np.int64)              # [F,3]


# ========= 2. 最远点采样(FPS)：自动选 handle =========
def farthest_point_sampling(V, K, seed=105):
    """
    在顶点集 V 上做最远点采样，返回 K 个顶点索引。
    V: [N,3] np.float64
    K: 需要的 handle 数量
    """
    np.random.seed(seed)
    N = V.shape[0]
    # 随机选一个起始点
    first = np.random.randint(0, N)
    selected = [first]

    # 每个点到“当前已选集合”的最小距离
    dist2 = np.full(N, np.inf, dtype=np.float64)

    for _ in range(1, K):
        last = selected[-1]
        # 距离平方
        diff = V - V[last]
        d2 = np.sum(diff * diff, axis=1)
        dist2 = np.minimum(dist2, d2)
        # 选取当前最远的点
        next_id = int(np.argmax(dist2))
        selected.append(next_id)

    return np.asarray(selected, dtype=np.int64)


# 举例：选 16 个 handle
K = 500
b = farthest_point_sampling(V, K)
# b = np.array([i for i in range(5023)])

print("[FPS] sampled handle indices:", b)


# ========= 3. 计算 BBW =========
def compute_vertex_bbw(V, F, b):
    """
    安全调用 libigl bbw，确保所有参数合法。
    V: [N,3] float64
    F: [M,3] int64
    b: [K]   int64, 顶点索引数组（控制点）
    返回:
        W: [N,K] 顶点对每个 handle 的权重
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    assert b.ndim == 1, "b 必须是一维数组，即多个顶点索引"
    K = int(b.shape[0])
    assert K > 0, "必须至少选一个 handle 顶点"

    # 边界条件：每个 handle 对应一行 one-hot
    bc = np.eye(K, dtype=np.float64)        # [K,K]
    # 初始值 W0 给 0 就行
    W0 = np.zeros((V.shape[0], K), dtype=np.float64)

    print(f"[BBW] vertices={V.shape[0]}, faces={F.shape[0]}, handles={K}")

    # 直接对表面三角网格调用 bbw 是支持的（libigl Python bindings 支持 3D triangle meshes）:contentReference[oaicite:0]{index=0}
    try:
        # 注意：不同版本 libigl.bbw 签名略有差异：
        # 通常是 igl.bbw(V, F, b, bc, W0, partition_unity)
        W = igl.bbw(V, F, b, bc, W0, True)
    except Exception as e:
        print("igl.bbw 调用失败:", e)
        raise

    # 数值清理：避免 NaN / 负数
    W = np.nan_to_num(W, nan=0.0)
    W = np.clip(W, 0.0, 1.0)
    row_sum = W.sum(axis=1, keepdims=True) + 1e-12
    W /= row_sum

    return W


W = compute_vertex_bbw(V, F, b)  # [V,K]
print("[BBW] W shape:", W.shape)
print("[BBW] W min/max/mean:", W.min(), W.max(), W.mean())


# np.save("/mnt/data/lyl/codes/RGBAvatar/BBW/129.npy", W)
# 写成 npz 文件
np.savez("/mnt/data/lyl/codes/RGBAvatar/BBW/vertice_and_faces_500/5023_500.npz",vertex_bbw=W,handle_indices=b)

# ========= 4. 可视化某个 handle 的权重 =========
def visualize_weight_field(V, F, W, handle_id=0, 
                           save_dir="./bbw_vis", dpi=300,
                           elev=0, azim=-90):
    """
    保存某个 handle 的权重场为 PNG，支持自定义视角角度。
    """
    from mpl_toolkits.mplot3d import Axes3D

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bbw_handle_{handle_id:02d}.png")

    w = W[:, handle_id]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    tri = ax.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        linewidth=0.1,
        antialiased=True,
        shade=False,
        edgecolor='none',
        cmap='viridis'
    )
    tri.set_array(w)
    tri.autoscale()

    fig.colorbar(tri, ax=ax, shrink=0.5, aspect=10, label=f"weight of handle {handle_id}")
    ax.set_title(f"Handle {handle_id} (elev={elev}, azim={azim})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 自定义视角
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"[VIS] Saved to {save_path}")



def visualize_weight_field_with_light(
    V, F, W, handle_id=0,
    save_dir="./bbw_vis", dpi=300,
    elev=0, azim=-90,
    light_dir=np.array([-0.3, 0.6, 1.0])  # 光照方向（世界坐标系）
):
    """
    带简单光照的 BBW 可视化：
    - 颜色由权重决定
    - 明暗由法线和光照方向决定
    """

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bbw_handle_{handle_id:02d}_light.png")

    w = W[:, handle_id]

    # ---- 1. 权重归一化到 [0,1]，增强对比度 ----
    w_min, w_max = float(w.min()), float(w.max())
    if abs(w_max - w_min) < 1e-12:
        w_vis = np.zeros_like(w)
        print(f"[WARN] handle {handle_id} is almost constant: {w_min:.6f}")
    else:
        w_vis = (w - w_min) / (w_max - w_min)

    # ---- 2. 计算顶点法线（由三角面片平均得到）----
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    N = V.shape[0]

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # [F,3]
    # 防止零长度
    fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    face_normals /= fn_norm

    vert_normals = np.zeros((N, 3), dtype=np.float64)
    for i in range(3):
        vert_normals[F[:, i]] += face_normals
    vn_norm = np.linalg.norm(vert_normals, axis=1, keepdims=True) + 1e-12
    vert_normals /= vn_norm

    # ---- 3. 计算光照强度 ----
    light_dir = light_dir.astype(np.float64)
    light_dir /= (np.linalg.norm(light_dir) + 1e-12)
    
    intensity = np.dot(vert_normals, light_dir)   # [N]
    intensity = np.clip(intensity, 0.0, 1.0)      # 只要漫反射部分
    # 加一点环境光，避免背面全黑
    # intensity = 0.3 + 0.7 * intensity             # 映射到 [0.3, 1.0]
    intensity = 0.4 + 0.6 * intensity             # 映射到 [0.3, 1.0]

    # ---- 4. 颜色 = colormap(w_vis) * intensity ----
    cmap = cm.get_cmap("viridis")
    base_colors = cmap(w_vis)      # [N,4] RGBA
    lit_colors = base_colors.copy()
    lit_colors[:, :3] *= intensity[:, None]  # RGB 乘以亮度，alpha 不变

    # ---- 5. 绘制 ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    tri = ax.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        linewidth=0.1,
        antialiased=True,
        shade=True,          # 我们自己做光照，所以关掉自带的 shade
        edgecolor="none"
    )
    tri.set_facecolors(lit_colors)

    # colorbar 仍然用权重（而不是亮度）
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(w_vis)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10,
                 label=f"weight of handle {handle_id}")

    ax.set_title(f"Handle {handle_id} (elev={elev}, azim={azim})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[VIS] Saved with light to {save_path}")
# 可视化第 0 个 handle
# for i in range(360):  # 保存前4个 handle 的可视化结果
#     # visualize_weight_field(V, F, W, handle_id=i, save_dir="./bbw_vis", elev=90, azim=-90, dpi=300)
#     # visualize_weight_field_with_light(V, F, W, handle_id=i, save_dir="./bbw_vis", elev=90, azim=-90, dpi=300)
#     visualize_weight_field_with_light(V, F, W, handle_id=i, save_dir="./bbw_vis", elev=90, azim=-90, dpi=300)
    
    
    
1