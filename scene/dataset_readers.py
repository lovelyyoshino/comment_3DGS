#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# 这里是数据库读取的代码，对这块魔改其实可以完成其他传感器的介入，本身colmap存在有输入的图像数据，以及处理好的点云数据信息

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

"""
定义了相机信息的数据结构，包括相机的uid、旋转矩阵R、平移向量T、
    垂直视场角FovY、水平视场角FovX、图像image、图像路径image_path、图像名称image_name、图像宽度width、图像高度height
"""
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

"""
定义了场景信息的数据结构，包括基本点云point_cloud、训练相机列表train_cameras、
    测试相机列表test_cameras、nerf归一化参数nerf_normalization、点云路径ply_path
"""
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

"""
计算nerf归一化参数，包括平移参数translate和半径参数radius，目的是将点云缩放到一个合适的大小，完成偏移和缩放的操作
cam_info: 相机信息列表
"""
def getNerfppNorm(cam_info):
    """
    获取相机中心和对角线的长度
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)#将cam_centers中的元素按列堆叠
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)#计算cam_centers的均值，输出为列向量，这个列向量的元素是cam_centers中每一列的均值
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)#计算cam_centers中每个元素到center的距离
        diagonal = np.max(dist)# 计算所有相机中心点到center的距离的最大值作为diagonal，即对角线的长度
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:#遍历cam_info中的每一个元素
        W2C = getWorld2View2(cam.R, cam.T)#获取世界坐标系到相机坐标系的变换矩阵
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])#将C2W的前三行第四列加入cam_centers，当中就是在世界坐标系下的相机的位置

    center, diagonal = get_center_and_diag(cam_centers)#获取相机中心和对角线的长度
    radius = diagonal * 1.1 # 计算出半径radius为diagonal的1.1倍

    translate = -center #平移参数为-center

    return {"translate": translate, "radius": radius}

"""
读取Colmap相机信息,
cam_extrinsics: 为相机外参信息字典
cam_intrinsics: 为相机内参信息字典
images_folder: 为图像文件夹路径
"""
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):# 遍历每个相机的外参信息
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]#根据key获取相机的外参信息
        intr = cam_intrinsics[extr.camera_id]#根据extr.camera_id获取相机的内参信息
        height = intr.height
        width = intr.width

        uid = intr.id#相机的id
        R = np.transpose(qvec2rotmat(extr.qvec))#将四元数转换为旋转矩阵，并将其转置
        T = np.array(extr.tvec)#平移向量

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]#获取相机的焦距
            FovY = focal2fov(focal_length_x, height)#计算垂直视场角
            FovX = focal2fov(focal_length_x, width)#计算水平视场角
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))#获取图像路径
        image_name = os.path.basename(image_path).split(".")[0]#获取图像名称
        image = Image.open(image_path)#读取图像

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)#构建相机信息
        cam_infos.append(cam_info)#将相机信息加入到cam_infos中
    sys.stdout.write('\n')
    return cam_infos

"""
读取Ply文件中的点云数据
path: 存储路径
"""
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T#ply中的点云
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0#ply中的颜色
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T#ply中的法向量
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

"""
将点云数据存储为Ply文件
path: 存储路径
xyz: 点云坐标
rgb: 点云颜色
"""
def storePly(path, xyz, rgb):
    # 定义ply文件的数据类型
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)#使用0初始化法向量

    elements = np.empty(xyz.shape[0], dtype=dtype)#创建一个空的数组，数据类型为dtype
    attributes = np.concatenate((xyz, normals, rgb), axis=1)#将xyz、normals、rgb按列拼接
    elements[:] = list(map(tuple, attributes))#将attributes中的元素转换为元组

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

"""
读取Colmap场景信息
path: 场景路径
images: 图像文件夹路径
eval: 是否为评估
llffhold: LLFF参数
"""
def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")#相机外参文件
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")#相机内参文件
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)#这里存放的是colmap处理好的相机在世界坐标系下的位置
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images#判断是否设置默认的images文件夹
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))#读取相机信息
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)#按照图像名称排序

    if eval:#如果是评估
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]#训练集，每隔llffhold取一个
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)#计算nerf归一化参数，包括平移参数translate和半径参数radius，这个参数是为了将点云缩放到一个合适的大小，完成偏移和缩放的操作，个人感觉可以直接使用点云的包围盒

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)#读取点云数据
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

"""
从变换文件中读取相机信息
path: 场景路径
transformsfile: 变换文件
white_background: 是否为白色背景
extension: 图像文件扩展名
"""
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:#读取变换文件
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]#获取相机的水平视场角

        frames = contents["frames"]#获取帧信息
        for idx, frame in enumerate(frames):#遍历每一帧
            cam_name = os.path.join(path, frame["file_path"] + extension)#获取图像路径

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])#获取相机到世界坐标系的变换矩阵
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1#将相机坐标系转换为COLMAP坐标系

            # 求逆得到世界到相机的变换矩阵
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))#将图像转换为RGBA格式

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])#背景颜色

            norm_data = im_data / 255.0#归一化
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])#将图像转换为RGB格式
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")#将图像转换为RGB格式

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])#计算垂直视场角
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

"""
读取Nerf合成数据集信息
path: 路径
white_background: 是否为白色背景
eval: 是否为评估
extension: 图像文件扩展名
"""
def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)#获取nerf归一化参数

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}