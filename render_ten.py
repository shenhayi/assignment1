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
from starter.utils import get_device, get_mesh_renderer


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


def render_tetrahedron_gif(
    output_path="images/tetrahedron_rotate.gif",
    image_size=256,
    color=[0.7, 0.7, 1],
    num_frames=60,
    fps=15,
):
    """
    Renders a 360-degree rotation of the tetrahedron along a spherical trajectory and saves it as a GIF.

    Args:
        output_path (str): Path where the output GIF will be saved.
        image_size (int): The size of the rendered image.
        color (list): RGB color for the tetrahedron texture.
        num_frames (int): The number of frames in the GIF.
        fps (int): Frames per second for the GIF.
    """
    device = get_device()

    # Create the renderer (this sets up the rasterizer and shader)
    renderer = get_mesh_renderer(image_size=image_size)

    # Define the 4 vertices of the tetrahedron
    vertices = torch.tensor([[-1.0, -1.0, -1.0],  # Vertex 0
                             [1.0, -1.0, -1.0],   # Vertex 1
                             [0.0, 1.0, -1.0],    # Vertex 2
                             [0.0, 0.0, 1.0]],    # Vertex 3
                            dtype=torch.float32)

    # Define the faces of the tetrahedron (using vertex indices)
    faces = torch.tensor([[0, 1, 2],  # Face 0
                          [0, 1, 3],  # Face 1
                          [1, 2, 3],  # Face 2
                          [2, 0, 3]], # Face 3
                         dtype=torch.long)

    # Convert to batch dimensions (add a batch dimension)
    vertices = vertices.unsqueeze(0)  # (4, 3) -> (1, 4, 3)
    faces = faces.unsqueeze(0)        # (4, 3) -> (1, 4, 3)

    # Define the textures (single color texture)
    textures = torch.ones_like(vertices)  # (1, 4, 3)
    textures = textures * torch.tensor(color)  # (1, 4, 3)

    # Create the mesh
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures)
    )
    mesh = mesh.to(device)

    # Set up a point light located in front of the tetrahedron
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
        description="Render a 360-degree rotating tetrahedron GIF with a spherical viewpoint using PyTorch3D."
    )
    parser.add_argument("--output_path", type=str, default="images/tetrahedron_rotate.gif", help="Path for the output GIF.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the rendered image.")
    parser.add_argument("--num_frames", type=int, default=60, help="Number of frames in the GIF.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the GIF.")
    args = parser.parse_args()

    render_tetrahedron_gif(
        output_path=args.output_path,
        image_size=args.image_size,
        num_frames=args.num_frames,
        fps=args.fps,
    )
