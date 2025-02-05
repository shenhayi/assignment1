"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import math

from starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/textured_cow_y.jpg")
    
    R_relative_upf=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T_relative_upf=[0.5, -0.5, 0]
    
    R_relative_z=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T_relative_z=[0, 0, 3]
    
    R_relative_z_rotate=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    T_relative_z_rotate=[0, 0, 0]
    
    theta = math.radians(90)
    R_relative_y_rotate=[[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]]
    T_relative_y_rotate=[-3, 0, 3]
    
    args = parser.parse_args()
    plt.imsave(args.output_path, render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, R_relative=R_relative_y_rotate, T_relative=T_relative_y_rotate))
