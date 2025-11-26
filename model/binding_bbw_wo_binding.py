# 去掉绑定
# test
from typing import Union, Optional
from abc import abstractmethod

import torch
from torch.nn import Parameter
import torch.nn.functional as F
import nvdiffrast.torch as dr

from diff_gaussian_rasterization import matrix_to_quaternion, quaternion_multiply, compute_face_tbn, fast_forward, mesh_binding
from submodules.flame import FLAME
from submodules.fuhead import FuHead
from utils import rgb2sh0, Struct
from utils import compute_face_tbn as compute_face_tbn_torch
from diff_renderer import compute_rast_info, GaussianAttributes
from .gaussian import GaussianModel

import numpy as np


def compute_orthogonality(vectors: torch.Tensor, p=2, norm=False):
    if norm:
        vectors = torch.nn.functional.normalize(vectors, dim=1)
    mat = vectors @ vectors.T
    triu_mat = torch.triu(mat, diagonal=1)
    return triu_mat.norm(p=p)


def Schmidt_orthogonalization(vectors: torch.Tensor) -> torch.Tensor:
    num_vectors, vector_dim = vectors.shape
    orthogonalized = torch.zeros_like(vectors)
    for i in range(num_vectors):
        # Start with the current vector
        v = vectors[i]
        
        # Subtract projections onto all previously orthogonalized vectors
        for j in range(i):
            u = orthogonalized[j]
            v -= torch.dot(v, u) / torch.dot(u, u) * u
        
        # Store the orthogonalized vector
        orthogonalized[i] = v
    return orthogonalized


def QR_orthogonalization(vectors: torch.Tensor) -> torch.Tensor:
    Q, R = torch.linalg.qr(vectors.T, mode='reduced')
    orthogonalized = Q.T * torch.norm(vectors, dim=1, keepdim=True)
    return orthogonalized


class BindingModel(GaussianModel):
    def __init__(self, 
        model_config: Struct,
        template_model: Union[FLAME, FuHead],
        glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]
    ):
        super().__init__(model_config)
        self.template_model = template_model
        self.template_uvs = self.template_model.uvs.to(torch.float32)
        self.template_faces = self.template_model.faces.to(torch.int32)
        self.template_uv_faces = self.template_model.uv_faces.to(torch.int32)
        self.face_uvs = self.template_uvs[self.template_uv_faces]
        self.glctx = glctx
        self.binding()

        # === 新增：加载 BBW 并计算每个高斯的 handle 权重 ===
        # bbw_data = np.load("/mnt/data/lyl/codes/RGBAvatar/BBW/vertice_and_faces/5083bbw.npz")
        bbw_data = np.load("/mnt/data/lyl/codes/RGBAvatar/BBW/vertice_and_faces_500/5083_500_bbw.npz")
        vertex_bbw = bbw_data["vertex_bbw"]          # [V,K]
        self.handle_indices = torch.from_numpy(
            bbw_data["handle_indices"]).long()       # [K]

        self.gaussian_bbw = compute_gaussian_bbw_weights(
            self.template_faces,
            self.binding_face_id,
            self.binding_face_bary,
            vertex_bbw
        ).to(device='cuda', dtype=torch.float32)     # [N,K]
        
    def binding(self):
        # binding gs with mesh triangle face
        face_uv, face_id = compute_rast_info( # FIXME: precision issue across different devices
            uvs=self.template_uvs,
            uv_faces=self.template_uv_faces,
            size=(self.model_config.tex_size, self.model_config.tex_size),
            glctx=self.glctx
        )
        face_id = face_id.reshape(-1)
        face_uv = face_uv.reshape(-1, 2)

        self.valid_binding_mask = face_id > 0
        face_uv = face_uv[self.valid_binding_mask]
        self.binding_face_id = face_id[self.valid_binding_mask > 0] - 1 # for which face does the gaussian binding.
        self.binding_face_bary = torch.cat( # for the barycentric of the binding face
            [face_uv, 1 - face_uv.sum(dim=-1, keepdim=True)], dim=-1)

    @property
    def num_gaussian(self):
        return self.binding_face_id.shape[0]

    def initialize(self):
        num_gaussian = self.num_gaussian
        print("Num gaussians:", num_gaussian)

        # initalize gaussian attributes
        xyz = torch.zeros([num_gaussian, 3], dtype=torch.float32, device='cuda')
        opacity = self.inv_opactity_act(torch.full([num_gaussian, 1], self.model_config.init_opacity, dtype=torch.float32, device='cuda'))
        scaling = self.inv_scaling_act(torch.full([num_gaussian, 3], self.model_config.init_scaling, dtype=torch.float32, device='cuda'))
        feature = torch.zeros([num_gaussian, 1, 3], dtype=torch.float32, device='cuda')
        rotation = torch.zeros([num_gaussian, 4], dtype=torch.float32, device='cuda')
        rotation[:, 0] = 1

        # initialize linear bases of gaussian attributes
        num_basis_blend = self.model_config.num_basis_blend if self.model_config.use_weight_proj else self.model_config.num_basis_in
        xyz_b = torch.zeros([num_basis_blend, num_gaussian, 3], dtype=torch.float32, device='cuda')
        feature_b = torch.zeros([num_basis_blend, num_gaussian, 1, 3], dtype=torch.float32, device='cuda')
        rotation_b = torch.zeros([num_basis_blend, num_gaussian, 4], dtype=torch.float32, device='cuda')

        # if gaussian used fast forward
        self.gs_initialized = torch.full([num_gaussian], False, dtype=torch.bool, device='cuda') # [N]

        # parameters
        self._xyz = Parameter(xyz.requires_grad_(True)) # [N, 3]
        self._opacity = Parameter(opacity.requires_grad_(True)) # [N, 1]
        self._scaling = Parameter(scaling.requires_grad_(True)) # [N, 3]
        self._rotation = Parameter(rotation.requires_grad_(True)) # [N, 4]
        self._feature_dc = Parameter(feature.requires_grad_(True)) # [N, 1, 3]

        self._xyz_b = Parameter(xyz_b.requires_grad_(True))
        self._feature_b = Parameter(feature_b.requires_grad_(True))
        self._rotation_b = Parameter(rotation_b.requires_grad_(True))

    @torch.no_grad()
    def fast_forward_torch(self, est_color, est_weight):
        est_color = torch.sum(est_color, dim=0)
        est_weight = torch.sum(est_weight, dim=0)
        est_weight_mask = est_weight > 0.1
        fast_forward_mask = torch.logical_and(est_weight_mask, ~self.gs_initialized)
        self.gs_initialized = torch.logical_or(self.gs_initialized, fast_forward_mask)
        fast_forward_indices = torch.nonzero(fast_forward_mask).squeeze(-1)
        self._feature_dc[fast_forward_indices, 0] = rgb2sh0(est_color[fast_forward_indices] / est_weight[fast_forward_indices, None])

    # 颜色初始化
    @torch.no_grad()
    def fast_forward(self, est_color, est_weight):
        est_color = torch.sum(est_color, dim=0)
        est_weight = torch.sum(est_weight, dim=0)
        fast_forward(
            0.1, # weight_threshold
            est_color,
            est_weight,
            self.gs_initialized,
            self._feature_dc
        )

    @abstractmethod
    def template_deform(self, deform_params) -> torch.Tensor:
        pass
    
    def gaussian_deform_batch(
        self,
        mesh_verts: torch.Tensor,                 # [B, V, 3]
        blend_weight: Optional[torch.Tensor]=None
    ) -> GaussianAttributes:
        """
        用 BBW 绑定：高斯位置 = handle 位置的加权和
        mesh_verts: 当前 deformed 网格顶点 [B,V,3]
        self.handle_indices: [K]
        self.gaussian_bbw: [N,K]
        """
        B, V, _ = mesh_verts.shape
        N = self.num_gaussian
        K = self.gaussian_bbw.shape[1]
        
        # 取出 handle 顶点在当前帧的坐标: [B,K,3]
        handle_pos = mesh_verts[:, self.handle_indices, :]  # [B,K,3]

        # 高斯的 BBW 权重: [N,K] -> [B,N,K]
        Wg = self.gaussian_bbw.to(mesh_verts.device)        # [N,K]
        Wg = Wg.unsqueeze(0).expand(B, -1, -1)              # [B,N,K]

        # 做线性组合: sum_k w_nk * handle_pos_k
        # handle_pos: [B,K,3] -> [B,1,K,3]
        HP = handle_pos.unsqueeze(1)                        # [B,1,K,3]
        # Wg: [B,N,K] -> [B,N,K,1]
        xyz = (Wg.unsqueeze(-1) * HP).sum(dim=2)            # [B,N,3]

        # 其他属性仍然用已有的 blendshape 模块
        gs = self.get_batch_attributes(B, blend_weight)     # [B,N,...]  
              
        rotation = gs.rotation                              # [B,N,4]

        return GaussianAttributes(xyz, gs.opacity, gs.scaling, rotation, gs.sh)
    
    def extract_texture(self):
        result = torch.zeros([self.model_config.tex_size, self.model_config.tex_size, 3], dtype=torch.float32, device='cuda').reshape(-1, 3)
        result[self.valid_binding_mask] = (self._feature_dc.squeeze(1) - 0.5) / 0.28209479177387814
        return result.reshape(self.model_config.tex_size, self.model_config.tex_size, 3)
    
    @torch.no_grad()
    def prune(self):
        not_optimized = ~self.gs_initialized
        self._opacity[not_optimized] = -9999
    
    @torch.no_grad()
    def clone(self):
        new_model = BindingModel(self.model_config, self.template_uvs, self.template_faces, self.template_uv_faces, self.glctx)
        # new_model.gs_initialized = self.gs_initialized.clone()
        new_model._xyz = self._xyz.clone()
        new_model._opacity = self._opacity.clone()
        new_model._scaling = self._scaling.clone()
        new_model._rotation = self._rotation.clone()
        new_model._feature_dc = self._feature_dc.clone()
        new_model._xyz_b = self._xyz_b.clone()
        new_model._rotation_b = self._rotation_b.clone()
        new_model._feature_b = self._feature_b.clone()
        new_model.weight_module.load_state_dict(self.weight_module.state_dict())
        return new_model
    
    def calc_orthogonality(self, p=2, norm=False):
        num_bases = self._xyz_b.shape[0]
        xyz_bases = self._xyz_b.reshape(num_bases, -1)
        rot_bases = self._rotation_b.reshape(num_bases, -1)
        rgb_bases = self._feature_b.reshape(num_bases, -1)
        xyz_orth = compute_orthogonality(xyz_bases, p=p, norm=norm)
        rot_orth = compute_orthogonality(rot_bases, p=p, norm=norm)
        rgb_orth = compute_orthogonality(rgb_bases, p=p, norm=norm)
        return xyz_orth, rot_orth, rgb_orth

    def orth_loss(self):
        xyz_orth, rot_orth, rgb_orth = self.calc_orthogonality(p=2, norm=True) # norm=True to balance losses of different attributes
        return xyz_orth + rot_orth + rgb_orth
    
    @torch.no_grad()
    def orthogonalize_Schmidt(self):
        num_bases = self._xyz_b.shape[0]
        xyz_bases = self._xyz_b.reshape(num_bases, -1)
        rot_bases = self._rotation_b.reshape(num_bases, -1)
        rgb_bases = self._feature_b.reshape(num_bases, -1)

        orth_xyz_b = Schmidt_orthogonalization(xyz_bases)
        orth_rot_b = Schmidt_orthogonalization(rot_bases)
        orth_rgb_b = Schmidt_orthogonalization(rgb_bases)

        self._xyz_b.data.copy_(orth_xyz_b.reshape(self._xyz_b.shape))
        self._rotation_b.data.copy_(orth_rot_b.reshape(self._rotation_b.shape))
        self._feature_b.data.copy_(orth_rgb_b.reshape(self._feature_b.shape))


    @torch.no_grad()
    def orthogonalize_QR(self):
        num_bases = self._xyz_b.shape[0]
        xyz_bases = self._xyz_b.reshape(num_bases, -1)
        rot_bases = self._rotation_b.reshape(num_bases, -1)
        rgb_bases = self._feature_b.reshape(num_bases, -1)

        orth_xyz_b = QR_orthogonalization(xyz_bases)
        orth_rot_b = QR_orthogonalization(rot_bases)
        orth_rgb_b = QR_orthogonalization(rgb_bases)

        self._xyz_b.data.copy_(orth_xyz_b.reshape(self._xyz_b.shape))
        self._rotation_b.data.copy_(orth_rot_b.reshape(self._rotation_b.shape))
        self._feature_b.data.copy_(orth_rgb_b.reshape(self._feature_b.shape))


class FLAMEBindingModel(BindingModel):
    def __init__(self, 
        model_config: Struct,
        flame_model: FLAME, 
        glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]
    ):
        super().__init__(model_config, flame_model, glctx)

    def template_deform(self, deform_params) -> torch.Tensor:
        verts = self.template_model(
            shape_params=deform_params.shape,
            expression_params=deform_params.exp,
            neck_pose_params=deform_params.neck_pose,
            jaw_pose_params=deform_params.jaw_pose,
            eye_pose_params=deform_params.eye_pose,
            eyelid_params=deform_params.eyelid_param
        )
        verts = torch.matmul(verts, deform_params.global_rot.transpose(-1, -2))
        verts += deform_params.global_transl.unsqueeze(1)
        return verts
        

class FuHeadBindingModel(BindingModel):
    def __init__(self, 
        model_config: Struct,
        fuhead_model: FuHead, 
        glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]
    ):
        super().__init__(model_config, fuhead_model, glctx)

    def template_deform(self, 
        identity: torch.Tensor, # no batch dim
        expression: torch.Tensor,
        eye_rotation: torch.Tensor,
        global_rotation: torch.Tensor,
        translation: torch.Tensor
    ) -> torch.Tensor:
        verts = self.template_model(identity, expression, eye_rotation)
        verts = torch.matmul(verts, global_rotation.transpose(-1, -2))
        verts += translation.unsqueeze(1)
        return verts


def compute_gaussian_bbw_weights(
        template_faces: torch.Tensor,      # [F,3] long 用顶点id表示的面片
        binding_face_id: torch.Tensor,     # [N] 每个高斯绑定的faceid
        binding_face_bary: torch.Tensor,   # [N,3] 高斯的绑定权重
        vertex_bbw: np.ndarray             # [V,K], numpy 顶点的bbw
    ) -> torch.Tensor:
    """
    把顶点 BBW 权重插值到每个 Gaussian 上：
    vertex_bbw[v,k]  -> gaussian_bbw[n,k]
    """
    if isinstance(template_faces, torch.Tensor):
        F = template_faces.cpu().numpy()
    else:
        F = template_faces
    face_ids = binding_face_id.cpu().numpy()
    bary = binding_face_bary.cpu().numpy()

    V2H = vertex_bbw  # [V,K]
    N = face_ids.shape[0]
    K = V2H.shape[1]
    gaussian_bbw = np.zeros((N, K), dtype=np.float32)

    for n in range(N):
        f_id = face_ids[n]
        v0, v1, v2 = F[f_id]  # 三角三个顶点索引
        w0, w1, w2 = bary[n]          # 重心系数
        gaussian_bbw[n] = (
            w0 * V2H[v0] +
            w1 * V2H[v1] +
            w2 * V2H[v2]
        )

    gaussian_bbw /= (gaussian_bbw.sum(axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(gaussian_bbw)  # [N,K]