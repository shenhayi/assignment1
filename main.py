import argparse
import imageio
import math
import matplotlib.pyplot as plt
import torch
import pytorch3d
import numpy as np
import mcubes
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def load_mesh(cow_path, color, device):
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    textures = torch.ones_like(vertices)  
    textures = textures * torch.tensor(color) 
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    return mesh.to(device)

def render_frame(mesh, renderer, cameras, lights):
    rend = renderer(mesh, cameras=cameras, lights=lights)
    image = rend[0, ..., :3].cpu().numpy()
    return image

# 1_1
def render_cow_gif(
    cow_path="data/cow.obj",
    output_path="images/cow_rotate.gif",
    image_size=256,
    color=[0.7, 0.7, 1],
    num_frames=60,
    fps=15,
):
    device = get_device()
    renderer = get_mesh_renderer(image_size=image_size)
    mesh = load_mesh(cow_path, color, device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = []  

    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        distance = 3.0  
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        image = render_frame(mesh, renderer, cameras, lights)
        images.append((image * 255).astype("uint8"))
    duration = 1 / fps
    imageio.mimsave(output_path, images, duration=duration, loop=0)

# 1_2
def dolly_zoom(
    image_size=256,
    num_frames=30,
    duration=3,
    output_file="images/dolly.gif",
):
    device = get_device()
    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)

    renders = []
    for fov in tqdm(fovs):
        distance = 30  
        T = [[0, 0, 150/fov]]  
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy() 
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop=0)

# 2_1 
def render_tetrahedron_gif(
    output_path="images/tetrahedron_rotate.gif",
    image_size=256,
    color=[0.7, 0.7, 1],
    num_frames=60,
    fps=15,
):
    device = get_device()
    renderer = get_mesh_renderer(image_size=image_size)
    vertices = torch.tensor([[-1.0, -1.0, -1.0],  # vertex 0
                             [1.0, -1.0, -1.0],   # vertex 1
                             [0.0, 1.0, -1.0],    # vertex 2
                             [0.0, 0.0, 1.0]],    # vertex 3
                            dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2],  # face 0
                          [0, 1, 3],  # face 1
                          [1, 2, 3],  # face 2
                          [2, 0, 3]], # face 3
                         dtype=torch.long)

    vertices = vertices.unsqueeze(0)  # (4, 3) -> (1, 4, 3)
    faces = faces.unsqueeze(0)        # (4, 3) -> (1, 4, 3)
    textures = torch.ones_like(vertices)  # (1, 4, 3)
    textures = textures * torch.tensor(color)  # (1, 4, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures)
    )
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = []  

   
    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        distance = 5.0  
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        image = render_frame(mesh, renderer, cameras, lights)
        images.append((image * 255).astype("uint8"))

    duration = 1 / fps
    imageio.mimsave(output_path, images, duration=duration, loop=0)

# 2_2  
def render_cube_gif(
    output_path="images/cube_rotate.gif",
    image_size=256,
    color=[0.7, 0.7, 1],
    num_frames=60,
    fps=15,
):
    device = get_device()
    renderer = get_mesh_renderer(image_size=image_size)
    vertices = torch.tensor([
        [-1.0, -1.0, -1.0],  # vertex 0
        [1.0, -1.0, -1.0],   # vertex 1
        [1.0, 1.0, -1.0],    # vertex 2
        [-1.0, 1.0, -1.0],   # vertex 3
        [-1.0, -1.0, 1.0],   # vertex 4
        [1.0, -1.0, 1.0],    # vertex 5
        [1.0, 1.0, 1.0],     # vertex 6
        [-1.0, 1.0, 1.0],    # vertex 7
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # face 0
        [4, 5, 6], [4, 6, 7],  # face 1
        [0, 1, 5], [0, 5, 4],  # face 2
        [1, 2, 6], [1, 6, 5],  # face 3
        [2, 3, 7], [2, 7, 6],  # face 4
        [3, 0, 4], [3, 4, 7],  # face 5
    ], dtype=torch.long)

    vertices = vertices.unsqueeze(0)  # (8, 3) -> (1, 8, 3)
    faces = faces.unsqueeze(0)        # (12, 3) -> (1, 12, 3)
    textures = torch.ones_like(vertices)  # (1, 8, 3)
    textures = textures * torch.tensor(color)  # (1, 8, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures)
    )
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = []  

    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        distance = 5.0  
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        image = render_frame(mesh, renderer, cameras, lights)
        images.append((image * 255).astype("uint8"))
    duration = 1 / fps
    imageio.mimsave(output_path, images, duration=duration, loop=0)

# 3    
def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    num_frames=60,
):
    device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    verts = meshes.verts_packed() 
    coords = verts[:, 2]
    z_min = coords.min()
    z_max = coords.max()

    if z_max == z_min:
        alpha = torch.zeros_like(coords)
    else:
        alpha = (coords - z_min) / (z_max - z_min)
    alpha = alpha.unsqueeze(1)  
    color1 = torch.tensor([0.0, 0.5, 0.8], device=device)  # red + green + blue
    color2 = torch.tensor([0.8, 0.5, 0.0], device=device)  
    verts_rgb = (1 - alpha) * color1 + alpha * color2

    
    meshes.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb.unsqueeze(0))
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = []
    
    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        distance = 3.0
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        rendered_image = renderer(meshes, cameras=cameras, lights=lights)
        frame = rendered_image[0, ..., :3].cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        images.append(frame)
    
    duration = 1 / 15
    imageio.mimsave("images/retextured_cow.gif", images, duration=duration, loop=0)
    
def render_textured_cow_trans(
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

# 5_2_1
def render_torus_gif(
    image_size=256,
    num_samples=200,
    num_frames=60,
    fps=15,
    output_path="images/torus_rotate.gif",
):
    device = get_device()
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

    renderer = pytorch3d.renderer.get_points_renderer(image_size=image_size, device=device)
    images = []  
    distance = 3.0  

    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 30 * math.sin(2 * math.pi * i / num_frames)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        image = renderer(torus_point_cloud, cameras=cameras)
        frame = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        images.append(frame)

    duration = 1 / fps  
    imageio.mimsave(output_path, images, duration=duration, loop=0)

# 5_2_2
def render_mobius_strip_gif(image_size=256, num_samples=1000, num_frames=60,fps=15):
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

    renderer = pytorch3d.renderer.get_points_renderer(image_size=image_size, device=device)
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

# 5_3_1
def render_torus_gif_implicit(
    image_size=256,
    voxel_size=64,
    num_frames=60,
    fps=15,
    output_path="images/torus_implicit_rotate.gif",
):
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
  
    imageio.mimsave(output_path, images, fps=fps, loop=0)

# 5_3_2
def render_octahedron_gif_implicit(
    image_size=512,
    voxel_size=128,
    num_frames=60,
    fps=15,
    output_path="images/octahedron_rotate.gif",
):
    device = get_device()
    grid_min = -2.0
    grid_max = 2.0
    
    X, Y, Z = torch.meshgrid(
        [torch.linspace(grid_min, grid_max, voxel_size)] * 3,
        indexing='ij'
    )
    
    a = 1.2  
    smoothness = 0.8  
    voxels = (X.abs()**a + Y.abs()**a + Z.abs()**a)**(1/a) - smoothness
    vertices, faces = mcubes.marching_cubes(voxels.numpy(), 0)
    vertices = torch.tensor(vertices).float()
    vertices = (vertices / (voxel_size-1)) * (grid_max - grid_min) + grid_min
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0).to(device))

    mesh = pytorch3d.structures.Meshes(
        verts=[vertices.to(device)],
        faces=[torch.tensor(faces.astype(np.int64)).to(device)],
        textures=textures
    )

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 3.0, -3.0]],
        diffuse_color=((0.8, 0.8, 0.8),),
        specular_color=((0.4, 0.4, 0.4),),
        device=device
    )
    images = []
    camera_distance = 4.0

    for i in range(num_frames):
        azim = 360 * i / num_frames
        elev = 20 * math.sin(2 * math.pi * i / num_frames)
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=camera_distance, 
            elev=elev, 
            azim=azim, 
            device=device
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, 
            T=T, 
            fov=45,
            device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype(np.uint8))
        
    imageio.mimsave(output_path, images, fps=fps, loop=0)

def render_octahedron_dolly_zoom(
    image_size=512,
    voxel_size=128,
    num_frames=60,
    fps=15,
    output_path="images/octahedron_dolly_zoom.gif",
):
    device = get_device()
    grid_min = -2.0
    grid_max = 2.0
    X, Y, Z = torch.meshgrid(
        [torch.linspace(grid_min, grid_max, voxel_size)] * 3,
        indexing='ij'
    )
    
    a = 1.2
    smoothness = 0.8
    voxels = (X.abs()**a + Y.abs()**a + Z.abs()**a)**(1/a) - smoothness

    vertices, faces = mcubes.marching_cubes(voxels.numpy(), 0)
    vertices = torch.tensor(vertices).float()
    vertices = (vertices / (voxel_size-1)) * (grid_max - grid_min) + grid_min
    
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0).to(device))

    mesh = pytorch3d.structures.Meshes(
        verts=[vertices.to(device)],
        faces=[torch.tensor(faces.astype(np.int64)).to(device)],
        textures=textures
    )

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 3.0, -3.0]],
        diffuse_color=((0.8, 0.8, 0.8),),
        specular_color=((0.4, 0.4, 0.4),),
        device=device
    )

    start_fov = 120.0
    end_fov = 5.0     
    fovs = torch.linspace(start_fov, end_fov, num_frames)
    base_distance = 5.0 

    images = []
    
    for i in tqdm(range(num_frames)):
        current_fov = fovs[i]
        distance = base_distance * (start_fov / current_fov)
        azim = 360 * i / num_frames
        elev = 20 * math.sin(2 * math.pi * i / num_frames)
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=distance,
            elev=elev,
            azim=azim,
            device=device
        )
        
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=current_fov,
            device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype(np.uint8))
    imageio.mimsave(output_path, images, fps=fps, loop=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="1_1",
        choices=["1_1","1_2","2_1","2_2", "3", "4_1", "4_2", "4_3", "4_4", "5_1", "5_2_1", "5_2_2", "5_3_1", "5_3_2", "6"],
    )
    
    R_relative_upf=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T_relative_upf=[0.5, -0.5, 0]
    
    R_relative_z=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T_relative_z=[0, 0, 3]
    
    R_relative_z_rotate=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    T_relative_z_rotate=[0, 0, 0]
    
    theta = math.radians(90)
    R_relative_y_rotate=[[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]]
    T_relative_y_rotate=[-3, 0, 3]
    
    parser.add_argument("--cow_path", type=str, default="data/cow.obj", help="Path to the cow OBJ file.")
    args = parser.parse_args()
    if args.render == "1_1":
        render_cow_gif()
    elif args.render == "1_2":
        dolly_zoom()
    elif args.render == "2_1":
        render_tetrahedron_gif()
    elif args.render == "2_2":
        render_cube_gif()
    elif args.render == "3":
        render_textured_cow()
    elif args.render == "4_1":
        plt.imsave("images/textured_cow_z_rotate.jpg", render_textured_cow_trans(cow_path=args.cow_path, R_relative=R_relative_z_rotate, T_relative=T_relative_z_rotate))
    elif args.render == "4_2":
        plt.imsave("images/textured_cow_z.jpg", render_textured_cow_trans(cow_path=args.cow_path, R_relative=R_relative_z, T_relative=T_relative_z))
    elif args.render == "4_3":
        plt.imsave("images/textured_cow_upf.jpg", render_textured_cow_trans(cow_path=args.cow_path, R_relative=R_relative_upf, T_relative=T_relative_upf))
    elif args.render == "4_4":
        plt.imsave("images/textured_cow_y_rotate.jpg", render_textured_cow_trans(cow_path=args.cow_path, R_relative=R_relative_y_rotate, T_relative=T_relative_y_rotate))
    elif args.render == "5_1":
        render_textured_cow()
    elif args.render == "5_2_1":
        render_torus_gif()
    elif args.render == "5_2_2":
        render_mobius_strip_gif()
    elif args.render == "5_3_1":
        render_torus_gif_implicit()
    elif args.render == "5_3_2":
        render_octahedron_gif_implicit()
    elif args.render == "6":
        render_octahedron_dolly_zoom()
        
