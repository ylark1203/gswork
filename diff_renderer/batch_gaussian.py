from typing import Optional

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, BatchGaussianRasterizer, GaussianRasterizer
from camera import Camera
from . import GaussianAttributes


class BatchGaussianRenderer:
    def __init__(self,
        bg_color: torch.Tensor,
        static_camera: Camera,
        num_gaussians: int,
        batch_size: int,
        sh_degree: int = 0
    ) -> None:
        tanfovx = math.tan(static_camera.fov_x * 0.5)
        tanfovy = math.tan(static_camera.fov_y * 0.5)

        viewmatrix = static_camera.get_w2v
        projmatrix = static_camera.get_full_proj
        raster_settings = GaussianRasterizationSettings(
            image_height=int(static_camera.height),
            image_width=int(static_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix.transpose(0, 1).contiguous().cuda(),
            projmatrix=projmatrix.transpose(0, 1).contiguous().cuda(),
            sh_degree=sh_degree,
            campos=static_camera.get_pos.contiguous().cuda(),
            prefiltered=False,
            debug=False
        )

        self.batch_size = batch_size
        self.rasterizer = BatchGaussianRasterizer(
            max_gaussian_size=num_gaussians,
            max_batch_size=batch_size,
            raster_settings=raster_settings
        )


    # def render(self,
    #     bg_color: torch.Tensor,
    #     gs: GaussianAttributes,
    #     target_image: Optional[torch.Tensor] = None
    # ):
    # def render( # legacy
    #     self,
    #     bg_color: torch.Tensor,
    #     gs: GaussianAttributes,
    #     covarience: torch.Tensor,
    #     camera: Camera,
    #     target_image: torch.Tensor = None,
    #     sh_degree: int = 0,
    #     scaling_modifier: float = 1.0
    # ) -> dict[str, torch.Tensor]:
    #     screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device) + 0
    #     if screenspace_points.requires_grad: # requires_grad == False when inference
    #         screenspace_points.retain_grad()
        
    #     self.rasterizer.raster_settings.bg = bg_color
    #     gs.xyz = gs.xyz.contiguous()
    #     gs.sh = gs.sh.contiguous()
    #     gs.opacity = gs.opacity.contiguous()
    #     gs.scaling = gs.scaling.contiguous()
    #     gs.rotation = gs.rotation.contiguous()
    #     if target_image is not None:
    #         target_image = target_image.contiguous()

    #     color, alpha, est_color, est_weight, radii = self.rasterizer(
    #         means3D=gs.xyz,
    #         means2D=screenspace_points, 
    #         opacities=gs.opacity, 
    #         shs=gs.sh, 
    #         scales=gs.scaling, 
    #         rotations=gs.rotation,
    #         target_image=target_image
    #     )

    #     return {
    #         "color": color, 
    #         "alpha": alpha, 
    #         "est_color": est_color, 
    #         "est_weight": est_weight, 
    #         "radii": radii
    #     }
    def render( # legacy
        self,
        bg_color: torch.Tensor,
        gs: GaussianAttributes,
        covarience: torch.Tensor,
        camera: Camera,
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
                bg=bg_color[i],
                scale_modifier=scaling_modifier,
                viewmatrix=viewmatrix.transpose(0, 1).contiguous().cuda(),
                projmatrix=projmatrix.transpose(0, 1).contiguous().cuda(),
                sh_degree=sh_degree,
                campos=camera.get_pos.contiguous().cuda(),
                prefiltered=False,
                debug=False
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            cov3D_precomp = covarience
            color, alpha, est_color, est_weight, radii = rasterizer(
                means3D=gs.xyz[i],
                means2D=screenspace_points[i],
                shs=gs.sh[i],
                colors_precomp=None,
                opacities=gs.opacity[i],
                # scales=gs.scaling[i],
                # rotations=gs.rotation[i],
                # cov3D_precomp=None,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_precomp[i],
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