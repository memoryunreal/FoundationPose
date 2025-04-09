#!/usr/bin/env python3

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # 或者 "osmesa", 如果环境只支持软件渲染

import sys
import imageio
import numpy as np
import trimesh
import pyrender

def render_obj_360(
    obj_path, 
    output_video='output_360.mp4', 
    steps=60, 
    distance_factor=2.0, 
    fps=30
):
    """
    加载一个 OBJ 模型并进行 360° 环绕渲染，将结果保存为 MP4 视频。

    参数：
    - obj_path: OBJ 文件路径
    - output_video: 输出视频文件名
    - steps: 旋转多少步，每步角度 = 360 / steps
    - distance_factor: 决定相机距离模型边界的倍数
    - fps: 输出视频的帧率
    """
    
    # 1. 加载 OBJ 作为 trimesh
    mesh = trimesh.load(obj_path)
    
    # 2. 创建 pyrender 场景
    scene = pyrender.Scene()
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh))
    
    # 3. 添加相机
    camera = pyrender.PerspectiveCamera(yfov=np.radians(60.0), aspectRatio=1.0)
    camera_node = scene.add(camera, pose=np.eye(4))
    
    # 4. 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=np.eye(4))
    
    # 5. 创建离屏渲染器
    r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
    
    # 6. 根据模型大小，确定相机距离
    extents = mesh.extents  # (x_range, y_range, z_range)
    max_extent = np.max(extents)
    distance = distance_factor * max_extent
    
    # 7. 逐帧渲染
    frames = []
    for i in range(steps):
        angle_deg = 360.0 * i / steps
        angle_rad = np.radians(angle_deg)
        
        # 计算相机位置 (绕 Y 轴旋转)
        x = distance * np.sin(angle_rad)
        z = distance * np.cos(angle_rad)
        eye = np.array([x, 0.0, z])
        
        # look_at：eye -> target; up=[0,1,0]
        camera_pose = trimesh.transformations.look_at(
            eye=eye,
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
        
        # 更新相机姿态
        scene.set_pose(camera_node, camera_pose)
        
        # 渲染当前视图
        color, depth = r.render(scene)
        frames.append(color)
    
    # 8. 保存视频
    imageio.mimsave(output_video, frames, fps=fps)
    print(f"✅ 已成功生成 360° 环绕视频: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python render_obj_360.py your_model.obj [output.mp4]")
        sys.exit(1)
    
    obj_file = sys.argv[1]
    if len(sys.argv) > 2:
        out_file = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(obj_file))[0]
        out_file = f"{base_name}_360.mp4"
    
    render_obj_360(obj_file, out_file)
