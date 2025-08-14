import os
import sys
import yaml
import torch
from tqdm import tqdm
import nvdiffrast.torch as dr
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from model import BindingModel, FLAMEBindingModel
from dataset import NeRSembleGADataset
from submodules.flame import FLAME, FlameConfig
from model.mv_reconstruction import render_gs_batch
from utils import Struct


def save_image(image_data, output_img_path, index, alpha=False):
    if alpha: Image.fromarray(image_data).save(os.path.join(output_img_path, "{:0>5d}.png".format(index)))
    else: Image.fromarray(image_data).save(os.path.join(output_img_path, "{:0>5d}.jpg".format(index)))


def render(gaussian_model: BindingModel, dataset: NeRSembleGADataset, bg_color: torch.Tensor, alpha: bool, output_path):
    output_img_path = os.path.join(output_path, "render_image")
    os.makedirs(output_img_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    for cam_name in list(set(dataset.camera_names)):
        tmp_dir = os.path.join(output_img_path, cam_name)
        os.makedirs(tmp_dir, exist_ok=True)

    i = 0
    progress_bar = tqdm(range(len(dataset)), desc="Rendering")
    with ThreadPoolExecutor() as executor:
        for data in dataloader:
            mesh = data['mesh'].cuda()
            blend_weight = data['blend_weight'].cuda()
            cam_view_mat = data['cam_view_mat'].cuda()
            cam_proj_mat = data['cam_proj_mat'].cuda()
            cam_position = data['cam_position'].cuda()
            frame_index = data['frame_index']
            camera_name = data['camera_name']

            bs = mesh.shape[0]
            gaussian = gaussian_model.gaussian_deform_batch(mesh, blend_weight)
            render_pkg = render_gs_batch(
                dataset.fov_x, dataset.fov_y,
                dataset.image_width, dataset.image_height,
                cam_view_mat, cam_proj_mat, cam_position, 
                bg_color.unsqueeze(0).repeat(bs, 1), gaussian
            )
            if alpha: image = torch.cat([render_pkg["color"], render_pkg["alpha"]], dim=1)
            else: image = render_pkg["color"]

            image = torch.clamp(image, 0.0, 1.0)
            image = (image.permute(0, 2, 3, 1) * 255.0).to(dtype=torch.uint8, device='cpu').numpy()
            for j in range(bs): 
                output_dir = os.path.join(output_img_path, camera_name[j])
                executor.submit(save_image, image[j], output_dir, frame_index[j], alpha)
            i += bs
            progress_bar.update(bs)
    progress_bar.close()
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--subject", type=str, default="074")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--work_name", type=str, required=True)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--alpha", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    glctx = dr.RasterizeGLContext()
    output_path = os.path.join(args.output_dir, args.subject, args.work_name)
    with open(os.path.join(output_path, "config.yaml")) as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_dir = config['data_dir']
    json_path = f'{dataset_dir}/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_{args.subject}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_val.json'

    if config['template_type'] == 'Flame':
        flame_config = FlameConfig()
        flame_config.dtype = torch.float32
        flame_config.flame_geom_path = 'data/FLAME2023/flame2023.pkl'
        flame_model = FLAME(flame_config).cuda()

        test_dataset = NeRSembleGADataset(flame_model, json_path, train=False)
        gaussian_model = FLAMEBindingModel(Struct(**config['model']), flame_model, glctx)
    else:
        raise NotImplementedError

    gaussian_model.load_ply(os.path.join(output_path, "model.ply"))

    if args.white_bg: bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device='cuda')
    else: bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')

    render(gaussian_model, test_dataset, bg_color, args.alpha, output_path)