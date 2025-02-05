"""
Usage:
    python retexturing.py --image_size 512
"""

import argparse
import matplotlib.pyplot as plt
import torch
import pytorch3d

from starter.utils import get_device, get_mesh_renderer
from pytorch3d.renderer import TexturesVertex, FoVPerspectiveCameras, look_at_view_transform


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
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
    color1 = torch.tensor([0.0, 0.0, 1.0], device=device)  # Blue
    color2 = torch.tensor([1.0, 0.0, 0.0], device=device)  # Red

    # Compute vertex colors by linear interpolation.
    verts_rgb = (1 - alpha) * color1 + alpha * color2

    # Assign the computed vertex colors as the mesh texture.
    meshes.textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))

    # Set up the camera using look_at_view_transform.
    # Here, we use dist=3, elev=0, azim=90 to get a side view.
    # Note: the texture gradient is still computed from the original z values.
    R, T = look_at_view_transform(dist=3, elev=0, azim=90, device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Set up a point light.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # Render the mesh and return the RGB image.
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj",
                        help="Path to the cow OBJ file.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for the render.")
    parser.add_argument("--output_path", type=str, default="images/retextured_cow.jpg",
                        help="Path to save the rendered image.")
    args = parser.parse_args()

    # Render the cow with the above settings.
    rendered_img = render_textured_cow(
        cow_path=args.cow_path,
        image_size=args.image_size,
    )

    # Save the rendered image.
    plt.imsave(args.output_path, rendered_img)
    print(f"Saved image to {args.output_path}")
