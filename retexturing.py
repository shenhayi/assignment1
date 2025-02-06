"""
Usage:
    python retexturing.py --image_size 512
"""

import argparse
import matplotlib.pyplot as plt
import torch
import pytorch3d
import math
import imageio
import numpy as np
from starter.utils import get_device, get_mesh_renderer
from pytorch3d.renderer import TexturesVertex, FoVPerspectiveCameras, look_at_view_transform


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    num_frames=60,
    device=None,
):
    """
    Loads the cow mesh, computes vertex colors based on a linear interpolation
    from blue (for the smallest z coordinate, i.e. the front) to red (for the largest
    z coordinate, i.e. the back), assigns these colors as a texture, and renders the
    mesh from a side view.
    """
    if device is None:
        device = get_device()

    # Load the mesh.
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    # Get vertex positions.
    verts = meshes.verts_packed()  # Shape: (V, 3)

    # Compute texture colors using the z-coordinate (front-to-back in original view).
    coords = verts[:, 2]
    z_min = coords.min()
    z_max = coords.max()

    # Compute interpolation parameter alpha for each vertex.
    if z_max == z_min:
        alpha = torch.zeros_like(coords)
    else:
        alpha = (coords - z_min) / (z_max - z_min)
    alpha = alpha.unsqueeze(1)  # Shape: (V, 1)

    # Define two colors:
    # color1: blue (for the front, i.e. smallest z)
    # color2: red (for the back, i.e. largest z)
    color1 = torch.tensor([0.0, 0.5, 0.8], device=device)  # red + green + blue
    color2 = torch.tensor([0.8, 0.5, 0.0], device=device)  

    # Compute vertex colors by linear interpolation.
    verts_rgb = (1 - alpha) * color1 + alpha * color2

    # Assign the computed vertex colors as the mesh texture.
    meshes.textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    # Set up a point light.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    images = []
    
    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        distance = 3.0
        
        R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        rendered_image = renderer(meshes, cameras=cameras, lights=lights)
        frame = rendered_image[0, ..., :3].cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        images.append(frame)
    
    duration = 1 / 15
    
    imageio.mimsave("images/retextured_cow.gif", images, duration=duration, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj",
                        help="Path to the cow OBJ file.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for the render.")
    parser.add_argument("--output_path", type=str, default="images/retextured_cow.jpg",
                        help="Path to save the rendered image.")
    parser.add_argument("--num_frames", type=int, default=60)
    args = parser.parse_args()

    render_textured_cow(
        cow_path=args.cow_path,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )
    
