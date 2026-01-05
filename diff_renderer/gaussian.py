import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, BatchGaussianRasterizer
from camera import Camera
from dataclasses import dataclass


@dataclass
class GaussianAttributes:
    xyz: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    sh: torch.Tensor
    # affine2: torch.Tensor
    cov3D: torch.Tensor = None

def cov3x3_to_6(cov):
    return torch.stack([cov[...,0,0], cov[...,0,1], cov[...,0,2],
                        cov[...,1,1], cov[...,1,2], cov[...,2,2]], dim=-1)

def render_gs(
    camera: Camera,
    bg_color: torch.Tensor,
    gs: GaussianAttributes,
    target_image: torch.Tensor = None,
    sh_degree: int = 0,
    scaling_modifier: float = 1.0
) -> dict[str, torch.Tensor]:
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device) + 0
    if screenspace_points.requires_grad: # requires_grad == False when inference
        screenspace_points.retain_grad()

    # Set up rasterization configuration
    tanfovx = math.tan(camera.fov_x * 0.5)
    tanfovy = math.tan(camera.fov_y * 0.5)

    viewmatrix = camera.get_w2v
    projmatrix = camera.get_full_proj
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix.transpose(0, 1).contiguous().to(device='cuda', non_blocking=True),
        projmatrix=projmatrix.transpose(0, 1).contiguous().to(device='cuda', non_blocking=True),
        sh_degree=sh_degree,
        campos=camera.get_pos.contiguous().to(device='cuda', non_blocking=True),
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image. 
    use_cov = hasattr(gs, "cov3D") and (gs.cov3D is not None)

    cov_precomp = gs.cov3D if use_cov else None
    scales = None if use_cov else gs.scaling
    rotations = None if use_cov else gs.rotation

    color, alpha, est_color, est_weight, radii = rasterizer(
        means3D=gs.xyz,
        means2D=screenspace_points,
        shs=gs.sh,
        colors_precomp=None,
        opacities=gs.opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov_precomp,
        target_image=target_image
    )

    return {"color": color, "alpha": alpha, "est_color": est_color, "est_weight": est_weight, "radii": radii}


def render_gs_batch( # legacy
    camera: Camera,
    bg_color: torch.Tensor,
    gs: GaussianAttributes,
    target_image: torch.Tensor = None,
    sh_degree: int = 0,
    scaling_modifier: float = 1.0
) -> dict[str, torch.Tensor]:

    screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device) + 0
    if screenspace_points.requires_grad: # requires_grad == False when inference
        screenspace_points.retain_grad()

    tanfovx = math.tan(camera.fov_x * 0.5)
    tanfovy = math.tan(camera.fov_y * 0.5)

    viewmatrix = camera.get_w2v
    projmatrix = camera.get_full_proj
    bg_color = bg_color.reshape(-1, 3)
    if bg_color.shape[0] == 1:
        bg_color = bg_color.repeat(gs.xyz.shape[0], 1)

    bs = gs.xyz.shape[0]
    color_list = [] 
    alpha_list = [] 
    est_color_list = [] 
    est_weight_list = [] 
    radii_list = []
    for i in range(bs):
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color[i].to(device='cuda:0', non_blocking=True),
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix.transpose(0, 1).contiguous().cuda(),
            projmatrix=projmatrix.transpose(0, 1).contiguous().cuda(),
            sh_degree=sh_degree,
            campos=camera.get_pos.contiguous().cuda(),
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # use_cov = hasattr(gs, "cov3D") and (gs.cov3D is not None)

        color, alpha, est_color, est_weight, radii = rasterizer(
            means3D=gs.xyz[i],
            means2D=screenspace_points[i],
            shs=gs.sh[i],
            colors_precomp=None,
            opacities=gs.opacity[i],
            # scales=None if use_cov else gs.scaling[i],
            # rotations=None if use_cov else gs.rotation[i],
            # cov3D_precomp=cov3x3_to_6(gs.cov3D[i]).to('cuda:0') if use_cov else None,
            scales=gs.scaling[i],
            rotations=gs.rotation[i],
            cov3D_precomp=None,
            target_image=target_image[i] if target_image is not None else None
        )


        color_list.append(color)
        alpha_list.append(alpha)
        est_color_list.append(est_color)
        est_weight_list.append(est_weight)
        radii_list.append(radii)

    return {
        "color": torch.stack(color_list), 
        "alpha": torch.stack(alpha_list), 
        "est_color": torch.stack(est_color_list), 
        "est_weight": torch.stack(est_weight_list), 
        "radii": torch.stack(radii_list)
    }