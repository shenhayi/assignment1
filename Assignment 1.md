# Assignment 1

## 1. Practicing with Cameras

### 1.1. 360-degree Renders 

![](/home/haoyus/16825/PS/assignment1/images/cow_rotate.gif)

### 1.2 Re-creating the Dolly Zoom

![](/home/haoyus/16825/PS/assignment1/images/dolly.gif)

## 2. Practicing with Meshes

### 2.1 Constructing a Tetrahedron

![](/home/haoyus/16825/PS/assignment1/images/tetrahedron_rotate.gif)

### 2.2 Constructing a Cube

![](/home/haoyus/16825/PS/assignment1/images/cube_rotate.gif)

## 3. Re-texturing a mesh

**color1 = (0.0, 0.5, 0.8)**  **light blue**

**color2 = (0.8, 0.5, 0.0)**  **light yellow**

![retextured_cow](/home/haoyus/16825/PS/assignment1/images/retextured_cow.gif)

## 4. Camera Transformations

### 4.1 Rotate along z-axis

**R_relative_z_rotate=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]**

**T_relative_z_rotate=[0, 0, 0]**

![textured_cow_z_rotate](/home/haoyus/16825/PS/assignment1/images/textured_cow_z_rotate.jpg)

### 4.2 Move along z-axis

**R_relative_z=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]**

**T_relative_z=[0, 0, 3]**

![textured_cow_z](/home/haoyus/16825/PS/assignment1/images/textured_cow_z.jpg)

### 4.3 Move along x-y plane

**R_relative_upf=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]**

**T_relative_upf=[0.5, -0.5, 0]**

![textured_cow_upf](/home/haoyus/16825/PS/assignment1/images/textured_cow_upf.jpg)

### 4.4 Rotate along y-axis

**theta = math.radians(90)**

**R_relative_y_rotate=[[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]]**

**T_relative_y_rotate=[-3, 0, 3]**

![textured_cow_y](/home/haoyus/16825/PS/assignment1/images/textured_cow_y.jpg)

## 5. Rendering Generic 3D Representations

### 5.1 Rendering Point Clouds from RGB-D Images (10 points)

![pointcloud](/home/haoyus/16825/PS/assignment1/images/pointcloud.gif)

### 5.2 Parametric Functions (10 + 5 points)

![torus_rotate](/home/haoyus/16825/PS/assignment1/images/torus_rotate.gif)

![mobius_strip_rotate](/home/haoyus/16825/PS/assignment1/images/mobius_strip_rotate.gif)

### 5.3 Implicit Surfaces (15 + 5 points)

![torus_implicit_rotate](/home/haoyus/16825/PS/assignment1/images/torus_implicit_rotate.gif)

![](/home/haoyus/16825/PS/assignment1/images/octahedron_rotate.gif)

## 6. Do Something Fun (10 points)

### octahedron rotates with dolly zoom

![](/home/haoyus/16825/PS/assignment1/images/octahedron_dolly_zoom.gif)