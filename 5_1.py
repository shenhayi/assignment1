import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    look_at_view_transform,
)

# Import functions from your utils.py file.
from starter.utils import get_device, get_points_renderer, unproject_depth_image

def load_rgbd_data(path="data/rgbd_data.pkl"):
    """
    Loads a dictionary containing the RGB-D data for two images.
    The dictionary is assumed to contain the following keys:
       "rgb0", "depth0", "mask0", "camera0",
       "rgb1", "depth1", "mask1", "camera1"
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def construct_point_clouds(data, device):
    """
    Constructs three point clouds:
      1. The point cloud for the first image.
      2. The point cloud for the second image.
      3. The union of the first two point clouds.
    
    The function uses the unproject_depth_image function from utils.py.
    """
    # Retrieve the data for the first image and push to the chosen device.
    rgb0 = data["rgb0"].to(device)     # Expected shape: (S, S, 3)
    depth0 = data["depth0"].to(device) # Expected shape: (S, S)
    mask0 = data["mask0"].to(device)   # Expected shape: (S, S)
    camera0 = data["camera0"].to(device)  # A PyTorch3D camera

    # Retrieve the data for the second image.
    rgb1 = data["rgb1"].to(device)
    depth1 = data["depth1"].to(device)
    mask1 = data["mask1"].to(device)
    camera1 = data["camera1"].to(device)

    # Unproject the images into point clouds.
    points0, rgba0 = unproject_depth_image(rgb0, mask0, depth0, camera0)
    points1, rgba1 = unproject_depth_image(rgb1, mask1, depth1, camera1)

    # Build PyTorch3D Pointclouds for each image.
    pcd0 = Pointclouds(points=[points0], features=[rgba0])
    pcd1 = Pointclouds(points=[points1], features=[rgba1])

    # Form the union by concatenating points and features.
    union_points = torch.cat([points0, points1], dim=0)
    union_rgba = torch.cat([rgba0, rgba1], dim=0)
    pcd_union = Pointclouds(points=[union_points], features=[union_rgba])

    return pcd0, pcd1, pcd_union


def visualize_point_cloud(pcd, device, title="Point Cloud", image_size=512):
    """
    Renders the given point cloud from several viewpoints.
    The viewpoints are defined by a camera placed 6 units from the origin
    and with azimuth angles equally spaced around 360°.
    """
    # Create a points renderer using your utility function.
    # (You can adjust the radius and background_color as needed.)
    renderer = get_points_renderer(image_size=image_size, device=device, radius=0.01)

    num_views = 6
    rendered_images = []

    # Loop over azimuth angles (here: 0°, 60°, 120°, 180°, 240°, 300°)
    for azim in np.linspace(0, 360, num_views, endpoint=False):
        # Set a fixed distance and elevation (you may modify these values).
        R, T = look_at_view_transform(dist=6, elev=0, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Update the renderer's camera.
        renderer.rasterizer.cameras = cameras

        # Render the point cloud.
        rend = renderer(pcd)
        # The rendered image has shape (1, H, W, 4); extract RGB channels.
        image = rend[0, ..., :3].cpu().numpy()
        rendered_images.append(image)

        # Display the rendered view.
        plt.figure()
        plt.imshow(image)
        plt.title(f"{title} (Azimuth: {azim:.1f}°)")
        plt.axis("off")
        plt.show()

    return rendered_images


def main():
    # Select the device.
    device = get_device()

    # Load the RGB-D data.
    data = load_rgbd_data("data/rgbd_data.pkl")

    # Construct the point clouds.
    pcd0, pcd1, pcd_union = construct_point_clouds(data, device)

    # Visualize each point cloud from multiple viewpoints.
    print("Visualizing point cloud from the first image...")
    visualize_point_cloud(pcd0, device, title="Point Cloud 1")

    print("Visualizing point cloud from the second image...")
    visualize_point_cloud(pcd1, device, title="Point Cloud 2")

    print("Visualizing the union of the two point clouds...")
    visualize_point_cloud(pcd_union, device, title="Union Point Cloud")


if __name__ == "__main__":
    main()
