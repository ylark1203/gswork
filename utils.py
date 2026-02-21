import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from typing import Tuple
from typing import Union
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        info = ["{}={}".format(k, v) for k, v in self.__dict__.items()]
        return "Struct({})".format(", ".join(info))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    window = window.to(device=img1.device, dtype=img1.dtype)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def quaternion_multiply(p: torch.Tensor, q: torch.Tensor):
    """
    Returns the product of two quaternions.
    Adapted from roma.

    Args:
        p, q (...x4 tensor, WXYZ convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, WXYZ convention).
    """
    vector = (
        p[..., None, 0] * q[..., 1:] + 
        q[..., None, 0] * p[..., 1:] +
        torch.cross(p[..., 1:], q[..., 1:], dim=-1)
    )
    last = p[..., 0] * q[..., 0] - torch.sum(p[..., 1:] * q[..., 1:], axis=-1)
    return torch.cat((last[..., None], vector), dim=-1)

def matrix_to_quaternion(R: torch.Tensor):
    """
    Converts rotation matrix to unit quaternion representation.
    Adapted from roma.

    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of unit quaternions (...x4 tensor, WXYZ convention).
    """
    batch_shape = R.shape[:-2]
    matrix = R.flatten(end_dim=-3) if len(batch_shape) > 0 else R.unsqueeze(0)
    num_rotations, D1, D2 = matrix.shape
    assert((D1, D2) == (3,3)), "Input should be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    ind1 = torch.nonzero(choices != 3, as_tuple=True)[0]
    ind2 = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    i = choices[ind1]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind1, i + 1] = 1 - decision_matrix[ind1, -1] + 2 * matrix[ind1, i, i]
    quat[ind1, j + 1] = matrix[ind1, j, i] + matrix[ind1, i, j]
    quat[ind1, k + 1] = matrix[ind1, k, i] + matrix[ind1, i, k]
    quat[ind1, 0] = matrix[ind1, k, j] - matrix[ind1, j, k]

    quat[ind2, 1] = matrix[ind2, 2, 1] - matrix[ind2, 1, 2]
    quat[ind2, 2] = matrix[ind2, 0, 2] - matrix[ind2, 2, 0]
    quat[ind2, 3] = matrix[ind2, 1, 0] - matrix[ind2, 0, 1]
    quat[ind2, 0] = 1 + decision_matrix[ind2, -1]

    quat = F.normalize(quat, dim=1)
    quat = quat.reshape(batch_shape + quat.shape[1:]) if len(batch_shape) > 0 else quat.squeeze(0)
    return quat

# @torch.compile # Found NVIDIA GeForce GTX 1080 Ti which is too old to be supported by the triton GPU compiler
def compute_face_tbn(face_verts: torch.Tensor, face_uvs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v0, v1, v2 = face_verts.unbind(-2)
    uv0, uv1, uv2 = face_uvs.unbind(-2)

    edge1 = v1 - v0
    edge2 = v2 - v0

    deltaUV1 = uv1 - uv0
    deltaUV2 = uv2 - uv0

    normal = torch.cross(edge1, edge2, dim=-1)
    area = normal.norm(dim=-1, keepdim=True) * 0.5

    f = 1.0 / (deltaUV1[..., 0] * deltaUV2[..., 1] - deltaUV2[..., 0] * deltaUV1[..., 1])
    f = f.unsqueeze(1)

    tangent = f * (deltaUV2[..., 1].unsqueeze(1) * edge1 - deltaUV1[..., 1].unsqueeze(1) * edge2)
    bitangent = f * (-deltaUV2[..., 0].unsqueeze(1) * edge1 + deltaUV1[..., 0].unsqueeze(1) * edge2)

    tbn = torch.stack([tangent, bitangent, normal], dim=-1) # [F, 3, 3]
    tbn = F.normalize(tbn, dim=-2) # normalize tbn in one step
    return tbn

def gather_vert_attributes(
    face_attrs: torch.Tensor, # [F, C]
    face_weights: torch.Tensor, # [F, 1]
    faces: torch.Tensor # [F, 3]
) -> torch.Tensor:
    num_verts = faces.max()
    vert_attrs = torch.zeros([num_verts + 1, face_attrs.shape[1]], dtype=face_attrs.dtype, device=face_attrs.device)
    weighted_face_attrs = face_attrs * face_weights

    vert_attrs = vert_attrs.index_add(0, faces[:, 0], weighted_face_attrs)
    vert_attrs = vert_attrs.index_add(0, faces[:, 1], weighted_face_attrs)
    vert_attrs = vert_attrs.index_add(0, faces[:, 2], weighted_face_attrs)
    return vert_attrs

def rgb2sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def flatten_model_params(model: torch.nn.Module):
    flat_params = []
    for param in model.state_dict().values():
        flat_params.append(param.view(-1))
    return torch.cat(flat_params)

def load_flattened_model_params(flat_params: torch.Tensor, model: torch.nn.Module):
    state_dict = model.state_dict()
    offset = 0
    for key, param in state_dict.items():
        numel = param.numel()
        new_param = flat_params[offset:offset+numel].view(param.size())
        state_dict[key].copy_(new_param)
        offset += numel
    model.load_state_dict(state_dict)

def model_size(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))

def indentity(x: torch.Tensor) -> torch.Tensor:
    return x


def smooth(x: np.ndarray, weight = 0.9):
    last = x[0]
    smoothed = []
    for point in x:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def average_rotation(R: np.ndarray):
    R_avg = np.mean(R, axis=0)
    U, _, Vt = np.linalg.svd(R_avg)
    R_avg_corrected = U @ Vt
    return R_avg_corrected

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], L.shape[1], 6), dtype=torch.float, device="cuda")

    uncertainty[:, :, 0] = L[:, :, 0, 0]
    uncertainty[:, :, 1] = L[:, :, 0, 1]
    uncertainty[:, :, 2] = L[:, :, 0, 2]
    uncertainty[:, :, 3] = L[:, :, 1, 1]
    uncertainty[:, :, 4] = L[:, :, 1, 2]
    uncertainty[:, :, 5] = L[:, :, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:, :, 0] * r[:, :, 0] + r[:, :, 1] * r[:, :, 1] + r[:, :, 2] * r[:, :, 2] + r[:, :, 3] * r[:, :, 3])

    q = r / norm[:, :, None]

    R = torch.zeros((q.size(0), q.size(1), 3, 3), device='cuda')

    r = q[:, :, 0]
    x = q[:, :, 1]
    y = q[:, :, 2]
    z = q[:, :, 3]

    R[:, :, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, :, 0, 1] = 2 * (x * y - r * z)
    R[:, :, 0, 2] = 2 * (x * z + r * y)
    R[:, :, 1, 0] = 2 * (x * y + r * z)
    R[:, :, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, :, 1, 2] = 2 * (y * z - r * x)
    R[:, :, 2, 0] = 2 * (x * z - r * y)
    R[:, :, 2, 1] = 2 * (y * z + r * x)
    R[:, :, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    B = s.shape[0]

    L = torch.zeros((B, s.shape[1], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, :, 0, 0] = s[:, :, 0]
    L[:, :, 1, 1] = s[:, :, 1]
    L[:, :, 2, 2] = s[:, :, 2]

    L = R @ L
    return L

def build_rotation_bg(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    q: [B, G, 4]  (w, x, y, z)  (your code uses r as w)
    return: R [B, G, 3, 3]
    """
    assert q.dim() == 3 and q.size(-1) == 4, f"q must be [B,G,4], got {q.shape}"
    # safer normalize
    q = F.normalize(q, dim=-1, eps=eps)  # unit quaternion

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.empty((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)

    # standard quat to rot (right-handed)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)

    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)

    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def strip_symmetric_bg(M: torch.Tensor) -> torch.Tensor:
    """
    M: [B,G,3,3] symmetric
    return: [B,G,6] in order (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    """
    assert M.shape[-2:] == (3, 3)
    out = torch.empty((*M.shape[:-2], 6), device=M.device, dtype=M.dtype)
    out[..., 0] = M[..., 0, 0]
    out[..., 1] = M[..., 0, 1]
    out[..., 2] = M[..., 0, 2]
    out[..., 3] = M[..., 1, 1]
    out[..., 4] = M[..., 1, 2]
    out[..., 5] = M[..., 2, 2]
    return out


# =========================================================
# A1-方案1：RS -> R S A  （加入 3 个剪切参数，表达相关项）
# =========================================================

def build_scaling_rotation_shear(
    s: torch.Tensor,
    q: torch.Tensor,
    shear: torch.Tensor,
    *,
    scaling_activation: str = "none",  # "none" | "exp" | "softplus"
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    s:     [B,G,3] scaling (sx,sy,sz)
    q:     [B,G,4] quaternion (w,x,y,z)
    shear: [B,G,3] (a01,a02,a12) -> upper-triangular shear matrix A:
            A = [[1, a01, a02],
                 [0,  1, a12],
                 [0,  0,  1]]
    return:
        L: [B,G,3,3] where L = R @ (S @ A)  (equivalently R S A)
    """
    assert s.shape[-1] == 3 and q.shape[-1] == 4 and shear.shape[-1] == 3
    B, G = s.shape[0], s.shape[1]
    device, dtype = s.device, s.dtype

    s_eff = s
    R = build_rotation_bg(q, eps=eps)  # [B,G,3,3]

    # Build S = diag(sx,sy,sz)
    S = torch.zeros((B, G, 3, 3), device=device, dtype=dtype)
    S[..., 0, 0] = s_eff[..., 0]
    S[..., 1, 1] = s_eff[..., 1]
    S[..., 2, 2] = s_eff[..., 2]

    # Build A (upper-triangular with ones on diagonal)
    a01, a02, a12 = shear[..., 0], shear[..., 1], shear[..., 2]
    A = torch.zeros((B, G, 3, 3), device=device, dtype=dtype)
    A[..., 0, 0] = 1.0
    A[..., 1, 1] = 1.0
    A[..., 2, 2] = 1.0
    A[..., 0, 1] = a01
    A[..., 0, 2] = a02
    A[..., 1, 2] = a12

    # L = R @ (S @ A)
    # (S@A) keeps SPD covariance when you later do Sigma = L L^T
    L = R @ (S @ A)
    return L

def build_covariance_from_s_q_shear(
    s: torch.Tensor,
    q: torch.Tensor,
    shear: torch.Tensor,
    *,
    scaling_modifier: float | torch.Tensor = 1.0,
    scaling_activation: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return symmetric 6D covariance representation [B,G,6].
    """
    # allow scaling_modifier as scalar or tensor broadcastable to [B,G,3]
    s_mod = s * scaling_modifier
    L = build_scaling_rotation_shear(
        s_mod, q, shear,
        scaling_activation=scaling_activation,
        eps=eps
    )
    Sigma = L @ L.transpose(-1, -2)  # [B,G,3,3] SPD
    return strip_symmetric_bg(Sigma)