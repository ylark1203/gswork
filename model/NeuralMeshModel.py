import torch
import torch.nn as nn
import numpy as np


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵。
    
    Args:
        quaternions: 形状为 [..., 4] 的张量。
                     假设四元数格式为 [w, x, y, z] (实部在前)。
                     **注意：输入前请确保四元数已经归一化 (normalized)。**
    
    Returns:
        形状为 [..., 3, 3] 的旋转矩阵。
    """
    
    # 1. 拆解四元数的分量
    # r (real part) = w
    # i, j, k (imaginary parts) = x, y, z
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    
    # 2. 计算中间变量 (根据公式)
    # 这里的计算是为了对应上面的矩阵公式
    two_s = 2.0  # 因为假设是归一化的，所以 s = 2/dot(q, q) = 2
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),  # Row 0, Col 0
            two_s * (i * j - k * r),      # Row 0, Col 1
            two_s * (i * k + j * r),      # Row 0, Col 2
            
            two_s * (i * j + k * r),      # Row 1, Col 0
            1 - two_s * (i * i + k * k),  # Row 1, Col 1
            two_s * (j * k - i * r),      # Row 1, Col 2
            
            two_s * (i * k - j * r),      # Row 2, Col 0
            two_s * (j * k + i * r),      # Row 2, Col 1
            1 - two_s * (i * i + j * j),  # Row 2, Col 2
        ),
        dim=-1,
    )
    
    # 3. 重塑形状为 [..., 3, 3]
    return o.reshape(quaternions.shape[:-1] + (3, 3))    
    
    
class NeuralMeshModel(nn.Module):
    def __init__(self, template_vertices, template_faces, latent_dim=128):
        """
        template_vertices: [V, 3] 初始的静态网格顶点 (也就是 FLAME 的零姿态形状)
        template_faces: [F, 3] 面片索引 (保持不变)
        latent_dim: 输入驱动向量的维度
        """
        super().__init__()
        
        # 1. 注册静态模板 (不可训练，或者设为 Parameter 可微调)
        self.register_buffer('template_vertices', template_vertices) 
        self.register_buffer('faces', template_faces)
        self.num_verts = template_vertices.shape[0]

        # 2. 变形解码器 (Deformation Decoder)
        # 作用：从 Latent Code 预测每个顶点的偏移量
        # 输出维度 = 顶点数 * 3
        self.offset_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_verts * 3) # 直接输出所有顶点的 xyz 偏移
        )
        
        # 初始化为 0，保证初始状态就是模板形状
        nn.init.zeros_(self.offset_decoder[-1].weight)
        nn.init.zeros_(self.offset_decoder[-1].bias)

        # 3. 全局姿态解码器 (Global Pose Decoder)
        # 作用：预测头部的整体旋转 (四元数) 和 平移
        self.pose_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4 + 3) # 4(Rotation Quaternion) + 3(Translation)
        )
        
        # 初始化旋转为单位四元数 [1,0,0,0]，平移为 0
        nn.init.zeros_(self.pose_decoder[-1].weight)
        nn.init.zeros_(self.pose_decoder[-1].bias)
        with torch.no_grad():
            self.pose_decoder[-1].bias[0] = 1.0 # quat w = 1

    def forward(self, latent_code):
        """
        latent_code: [B, latent_dim] (比如来自音频编码器)
        """
        B = latent_code.shape[0]
        
        # --- A. 计算局部非刚性变形 ---
        # [B, V*3] -> [B, V, 3]
        offsets = self.offset_decoder(latent_code).view(B, self.num_verts, 3)
        
        # 变形后的网格 (在局部坐标系下，类似于 FLAME 的 Canonical space)
        # [1, V, 3] + [B, V, 3]
        deformed_verts_local = self.template_vertices.unsqueeze(0) + offsets
        
        # --- B. 计算全局刚性变换 ---
        pose_params = self.pose_decoder(latent_code)
        rot_quat = pose_params[:, :4] # [B, 4]
        translation = pose_params[:, 4:] # [B, 3]
        
        # 归一化四元数
        rot_quat = torch.nn.functional.normalize(rot_quat, dim=-1)
        # 将四元数转为旋转矩阵 [B, 3, 3] (需引入一个辅助函数或 pytorch3d)
        rot_mat = quaternion_to_matrix(rot_quat) 
        
        # --- C. 应用全局变换 ---
        # X_world = R * X_local + T
        # [B, V, 3] = ([B, 3, 3] @ [B, V, 3]^T)^T + [B, 1, 3]
        deformed_verts_world = torch.bmm(deformed_verts_local, rot_mat.permute(0, 2, 1)) + translation.unsqueeze(1)
        
        return deformed_verts_world