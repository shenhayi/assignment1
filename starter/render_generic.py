"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle
import imageio
import os
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import math

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus(image_size=256, num_samples=200, device=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (1 + 0.5 * torch.cos(Theta)) * torch.cos(Phi)
    y = (1 + 0.5 * torch.cos(Theta)) * torch.sin(Phi)
    z = 0.5 * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(torus_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus_gif(
    image_size=256,
    num_samples=200,
    num_frames=60,
    fps=15,
    output_path="images/torus_rotate.gif",
    device=None,
):
    """
    Renders a 360-degree rotation of the torus as a point cloud along a spherical trajectory and saves it as a GIF.

    Args:
        image_size (int): The size of the rendered image.
        num_samples (int): Number of samples for the torus parametric grid.
        num_frames (int): Number of frames in the GIF.
        fps (int): Frames per second for the GIF.
        output_path (str): Path where the output GIF will be saved.
        device: Device to run the rendering on.
    """
    if device is None:
        device = get_device()

    # Generate the torus point cloud (using the same parametric sampling as in render_torus)
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta, indexing="ij")
    x = (1 + 0.5 * torch.cos(Theta)) * torch.cos(Phi)
    y = (1 + 0.5 * torch.cos(Theta)) * torch.sin(Phi)
    z = 0.5 * torch.sin(Theta)
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points.to(device)], features=[color.to(device)]
    )

    # Create a points renderer.
    renderer = get_points_renderer(image_size=image_size, device=device)

    images = []  # To store rendered frames
    distance = 3.0  # Distance of the camera from the torus

    # Generate frames along a 360-degree rotation.
    for i in range(num_frames):
        azim = 360 * i / num_frames
        # Optionally vary elevation sinusoidally between -30° and 30°
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        # Render the current frame.
        image = renderer(torus_point_cloud, cameras=cameras)
        # Convert the image to uint8.
        frame = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        images.append(frame)
        print(f"Rendered frame {i+1}/{num_frames}: azimuth {azim:.1f}°, elevation {elev:.1f}°")

    duration = 1 / fps  # Duration per frame in seconds
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    print(f"GIF saved to {output_path}")

def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def render_torus_gif_implicit(
    image_size=256,
    voxel_size=64,
    num_frames=60,
    fps=15,
    output_path="images/torus_implicit_rotate.gif",
    device=None,
):
    """
    Renders a 360-degree rotation of an implicitly defined torus using marching cubes
    and saves it as a GIF.

    Args:
        image_size (int): The size of the rendered image.
        voxel_size (int): Resolution of the voxel grid for marching cubes.
        num_frames (int): Number of frames in the GIF.
        fps (int): Frames per second for the GIF.
        output_path (str): Path where the output GIF will be saved.
        device: Device to run the rendering on.
    """
    if device is None:
        device = get_device()

    R = 1.0
    r = 0.5
    min_value = -2.0
    max_value = 2.0
    
    X, Y, Z = torch.meshgrid(
        [torch.linspace(min_value, max_value, voxel_size)] * 3,
        indexing='ij'
    )
    
    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels.numpy()), isovalue=0)
    
    vertices = torch.tensor(vertices).float()
    vertices = (vertices / (voxel_size-1)) * (max_value - min_value) + min_value
    
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(verts_features=textures.unsqueeze(0).to(device))

    faces = torch.tensor(faces.astype(np.int64)).to(device)
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices.to(device)], 
        faces=[faces], 
        textures=textures
    )

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)

    images = []
    distance = 5.0 

    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 30 * math.sin(2 * math.pi * i / num_frames) 
        
        R_cam, T_cam = pytorch3d.renderer.look_at_view_transform(
            dist=distance, 
            elev=elev, 
            azim=azim, 
            device=device
        )
        
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R_cam, 
            T=T_cam, 
            fov=60, 
            device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype(np.uint8))
        
        print(f"Rendered frame {i+1}/{num_frames}: azim={azim:.1f}°, elev={elev:.1f}°")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps, loop=0)
    print(f"GIF saved to {output_path}")
    
def render_mobius_strip(image_size=256, num_samples=1000, device=None):
    """
    Renders a Möbius strip using parametric sampling. Samples num_samples ** 2 points.
    """
    if device is None:
        device = get_device()

    u = torch.linspace(0, 2 * np.pi, num_samples)  
    v = torch.linspace(-0.5, 0.5, num_samples)     
    
    U, V = torch.meshgrid(u, v, indexing='ij')
    
    w = 0.2  # width
    x = (1 + w * V * torch.cos(U/2)) * torch.cos(U)
    y = (1 + w * V * torch.cos(U/2)) * torch.sin(U)
    z = w * V * torch.sin(U/2)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    
    color = (points - points.min()) / (points.max() - points.min())

    mobius_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points.to(device)], 
        features=[color.to(device)]
    )

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        T=[[0, 0, 3]],  
        device=device
    )
    renderer = get_points_renderer(image_size=image_size, device=device)
    

    rend = renderer(mobius_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_mobius_strip_gif(image_size=256, num_samples=1000, num_frames=60,fps=15, device=None):
    """
    Renders a Möbius strip using parametric sampling. Samples num_samples ** 2 points.
    """
    if device is None:
        device = get_device()

    u = torch.linspace(0, 2 * np.pi, num_samples)  
    v = torch.linspace(-0.5, 0.5, num_samples)     
    
    U, V = torch.meshgrid(u, v, indexing='ij')
    
    w = 0.5  # width
    x = (1 + w * V * torch.cos(U/2)) * torch.cos(U)
    y = (1 + w * V * torch.cos(U/2)) * torch.sin(U)
    z = w * V * torch.sin(U/2)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    
    color = (points - points.min()) / (points.max() - points.min())

    mobius_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points.to(device)], 
        features=[color.to(device)]
    )

    renderer = get_points_renderer(image_size=image_size, device=device)
    images = []
    distance = 3.0
    
    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        image = renderer(mobius_point_cloud, cameras=cameras)
        frame = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        images.append(frame)
    
    duration = 1 / fps
    imageio.mimsave("images/mobius_strip_rotate.gif", images, duration=duration, loop=0)

def render_mobius_strip_gif_implicit(
    image_size=256,
    voxel_size=128,
    num_frames=60,
    fps=15,
    output_path="images/mobius_implicit_rotate.gif",
    device=None,
):
    """
    Renders a Möbius strip using a corrected implicit equation and parameterization
    """
    if device is None:
        device = get_device()

    # Adjusted grid bounds for Mobius strip dimensions
    min_value = -2.5
    max_value = 2.5
    
    # Create 3D grid with higher resolution
    X, Y, Z = torch.meshgrid(
        [torch.linspace(min_value, max_value, voxel_size)] * 3,
        indexing='ij'
    )
    
    # Corrected implicit equation for Möbius strip
    R = 1.0  # Major radius
    w = 0.3  # Half-width parameter
    term1 = (X**2 + Y**2 + Z**2)**2
    term2 = 2*(X**2 + Y**2 + Z**2)*(R**2 - w**2 + X**2 + Y**2)
    term3 = (R**2 + w**2 - X**2 - Y**2)**2 - 4*R**2*(X**2 - Y**2) - 8*R*w*Y*Z
    voxels = term1 - term2 + term3

    # Extract surface using marching cubes
    vertices, faces = mcubes.marching_cubes(voxels.numpy(), 0)
    
    # Normalize vertices to world coordinates
    vertices = torch.tensor(vertices).float()
    vertices = (vertices / (voxel_size-1)) * (max_value - min_value) + min_value
    
    # Create vertex-based coloring
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0).to(device))

    # Create mesh with face normals
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices.to(device)],
        faces=[torch.tensor(faces.astype(np.int64)).to(device)],
        textures=textures
    )

    # Configure renderer with improved lighting
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 2.0, -2.0]],
        diffuse_color=((0.9, 0.9, 0.9),),
        specular_color=((0.3, 0.3, 0.3),),
        device=device
    )

    # Generate rotation animation
    images = []
    distance = 10.0  # Optimal viewing distance

    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 20 * math.sin(2 * math.pi * i / num_frames)  # Gentle elevation change
        
        R_cam, T_cam = pytorch3d.renderer.look_at_view_transform(
            dist=distance, 
            elev=elev, 
            azim=azim, 
            device=device
        )
        
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R_cam, 
            T=T_cam, 
            fov=50,  # Field of view adjustment
            device=device
        )

        # Render with anti-aliasing
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype(np.uint8))
        
        print(f"Rendered frame {i+1}/{num_frames}")

    # Save animated GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps, loop=0)
    print(f"GIF saved to {output_path}")
    
def render_octahedron_gif_implicit(
    image_size=512,
    voxel_size=128,
    num_frames=60,
    fps=15,
    output_path="images/octahedron_rotate.gif",
    device=None,
):
    """
    Renders an octahedron using implicit surface representation with marching cubes
    """
    if device is None:
        device = get_device()

    # 定义三维网格参数
    grid_min = -2.0
    grid_max = 2.0
    
    # 创建三维体素网格
    X, Y, Z = torch.meshgrid(
        [torch.linspace(grid_min, grid_max, voxel_size)] * 3,
        indexing='ij'
    )
    
    # 八面体隐式方程 (|x| + |y| + |z| = 1 的平滑版本)
    a = 1.2  # 形状参数
    smoothness = 0.8  # 平滑系数
    voxels = (X.abs()**a + Y.abs()**a + Z.abs()**a)**(1/a) - smoothness

    # 使用行进立方体提取表面
    vertices, faces = mcubes.marching_cubes(voxels.numpy(), 0)
    
    # 将顶点坐标标准化到世界坐标系
    vertices = torch.tensor(vertices).float()
    vertices = (vertices / (voxel_size-1)) * (grid_max - grid_min) + grid_min
    
    # 创建顶点颜色纹理
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0).to(device))

    # 构建网格对象
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices.to(device)],
        faces=[torch.tensor(faces.astype(np.int64)).to(device)],
        textures=textures
    )

    # 配置渲染器
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 3.0, -3.0]],
        diffuse_color=((0.8, 0.8, 0.8),),
        specular_color=((0.4, 0.4, 0.4),),
        device=device
    )

    # 生成动画帧
    images = []
    camera_distance = 4.0

    for i in range(num_frames):
        # 计算相机角度
        azim = 360 * i / num_frames
        elev = 20 * math.sin(2 * math.pi * i / num_frames)
        
        # 相机变换矩阵
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=camera_distance, 
            elev=elev, 
            azim=azim, 
            device=device
        )
        
        # 配置相机参数
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, 
            T=T, 
            fov=45,
            device=device
        )

        # 渲染当前帧
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype(np.uint8))
        
        print(f"Rendered frame {i+1}/{num_frames}")

    # 保存GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps, loop=0)
    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit", "torus_gif", "torus_implicit_gif","mobius_strip", "mobius_gif","mobius_implicit_gif", "octagedron_implicit_gif"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        image = render_torus(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    elif args.render == "torus_gif":
        image = render_torus_gif(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "torus_implicit_gif":
        image = render_torus_gif_implicit(image_size=args.image_size)
    elif args.render == "mobius_strip":
        image = render_mobius_strip(image_size=args.image_size)
    elif args.render == "mobius_gif":
        image = render_mobius_strip_gif(image_size=args.image_size)
    elif args.render == "mobius_implicit_gif":
        image = render_mobius_strip_gif_implicit(image_size=args.image_size)
    elif args.render == "octagedron_implicit_gif":
        image = render_octahedron_gif_implicit(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)

