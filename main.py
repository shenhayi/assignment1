"""
Sample code to render a cow from a spherical trajectory and save it as a GIF.

Usage:
    python main.py --cow_path data/cow.obj --output_path images/cow_rotate.gif --image_size 256 --num_frames 60 --fps 15
"""

import argparse
import imageio
import math
import matplotlib.pyplot as plt
import torch
import pytorch3d
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    look_at_view_transform,
)
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def load_mesh(cow_path, color, device):
    """
    Loads the cow mesh data and sets the vertex texture color.

    Args:
        cow_path (str): Path to the cow OBJ file.
        color (list): RGB color to apply to the mesh.
        device (torch.device): The device (CPU/GPU) to use.

    Returns:
        Meshes: A PyTorch3D Meshes object with the cow mesh.
    """
    # Load vertices and faces
    vertices, faces = load_cow_mesh(cow_path)
    # Add batch dimension: (N_v, 3) -> (1, N_v, 3)
    vertices = vertices.unsqueeze(0)
    # (N_f, 3) -> (1, N_f, 3)
    faces = faces.unsqueeze(0)
    # Create textures with the specified color for each vertex
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    # Create and return the Mesh object
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    return mesh.to(device)


def render_frame(mesh, renderer, cameras, lights):
    """
    Renders a single frame using the provided renderer, cameras, and lights.

    Args:
        mesh (Meshes): The mesh to render.
        renderer: The mesh renderer.
        cameras: The camera configuration.
        lights: The light configuration.

    Returns:
        np.array: The rendered image as an (H, W, 3) numpy array.
    """
    rend = renderer(mesh, cameras=cameras, lights=lights)
    # The rendered output has shape (B, H, W, 4); we take the first batch and only the RGB channels.
    image = rend[0, ..., :3].cpu().numpy()
    return image


def render_cow_gif(
    cow_path="data/cow.obj",
    output_path="images/cow_rotate.gif",
    image_size=256,
    color=[0.7, 0.7, 1],
    num_frames=60,
    fps=15,
):
    """
    Renders a 360-degree rotation of the cow along a spherical trajectory and saves it as a GIF.

    Args:
        cow_path (str): Path to the cow OBJ file.
        output_path (str): Path where the output GIF will be saved.
        image_size (int): The size of the rendered image.
        color (list): RGB color for the cow texture.
        num_frames (int): The number of frames in the GIF.
        fps (int): Frames per second for the GIF.
    """
    device = get_device()

    # Create the renderer (this sets up the rasterizer and shader)
    renderer = get_mesh_renderer(image_size=image_size)

    # Load the cow mesh with the specified texture color
    mesh = load_mesh(cow_path, color, device)

    # Set up a point light located in front of the cow
    lights = PointLights(location=[[0, 0, -3]], device=device)

    images = []  # List to store each rendered frame

    # Generate frames from 0 to 360 degrees (excluding the final duplicate frame)
    # The camera will move along a spherical path by varying both azimuth and elevation.
    for i in range(num_frames):
        # Compute azimuth linearly from 0 to 360 degrees
        azim = 360 * i / num_frames
        # Compute elevation to vary sinusoidally between -30° and +30°
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        distance = 3.0  # Distance from the scene center

        # Generate the camera transformation based on the current azimuth and elevation angles
        R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

        # Render the current frame
        image = render_frame(mesh, renderer, cameras, lights)
        images.append((image * 255).astype("uint8"))

        print(f"Rendered frame {i+1}/{num_frames}: azimuth {azim:.1f}°, elevation {elev:.1f}°")

    # Convert frames per second to duration per frame in seconds
    duration = 1 / fps

    # Save the images as a looping GIF
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    print(f"GIF saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a 360-degree rotating cow GIF with a spherical viewpoint using PyTorch3D."
    )
    parser.add_argument("--cow_path", type=str, default="data/cow.obj", help="Path to the cow OBJ file.")
    parser.add_argument("--output_path", type=str, default="images/cow_rotate.gif", help="Path for the output GIF.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the rendered image.")
    parser.add_argument("--num_frames", type=int, default=60, help="Number of frames in the GIF.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the GIF.")
    args = parser.parse_args()

    render_cow_gif(
        cow_path=args.cow_path,
        output_path=args.output_path,
        image_size=args.image_size,
        num_frames=args.num_frames,
        fps=args.fps,
    )
