"""
点云转换工具模块
负责将深度图像转换为3D点云，并应用坐标系变换
"""
import numpy as np
import open3d as o3d
import cv2


# 坐标系转换矩阵，用于统一OpenCV、Open3D和深度相机坐标系
COORD_TRANSFORM = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


class DepthToPointCloudConverter:
    """深度图像转点云计算类"""
    
    def __init__(self, fx, fy, cx, cy, depth_scale=1000.0):
        """初始化转换器参数"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    def convert(self, depth_map, min_dist=0.55, max_dist=8.0):
        """转换深度图到点云"""
        pcd = o3d.geometry.PointCloud()
        height, width = depth_map.shape
        z_meters = depth_map.astype(np.float32) / self.depth_scale

        # 创建有效深度掩码
        valid_mask = (z_meters >= min_dist) & (z_meters <= max_dist)
        v, u = np.indices((height, width))

        z_valid = z_meters[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        if len(z_valid) == 0:
            return pcd

        # 计算3D坐标
        x_valid = (u_valid - self.cx) * z_valid / self.fx
        y_valid = (v_valid - self.cy) * z_valid / self.fy
        points = np.stack((x_valid, y_valid, z_valid), axis=-1)

        # 应用坐标系变换
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        points_transformed_homogeneous = points_homogeneous @ COORD_TRANSFORM.T
        points = points_transformed_homogeneous[:, :3]

        # 简单着色
        norm_z = (z_valid - min_dist) / (max_dist - min_dist)
        colors_rgb = np.tile([0.5, 0.5, 0.5], (len(points), 1))  # 灰色

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        return pcd

    def convert_with_color(self, depth_map, color_image, min_dist=0.55, max_dist=8.0):
        """转换深度图和彩色图到彩色点云"""
        pcd = o3d.geometry.PointCloud()
        height, width = depth_map.shape
        z_meters = depth_map.astype(np.float32) / self.depth_scale

        # 创建有效深度掩码
        valid_mask = (z_meters >= min_dist) & (z_meters <= max_dist)
        v, u = np.indices((height, width))

        z_valid = z_meters[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        if len(z_valid) == 0:
            return pcd

        # 计算3D坐标
        x_valid = (u_valid - self.cx) * z_valid / self.fx
        y_valid = (v_valid - self.cy) * z_valid / self.fy
        points = np.stack((x_valid, y_valid, z_valid), axis=-1)

        # 应用坐标系变换
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        points_transformed_homogeneous = points_homogeneous @ COORD_TRANSFORM.T
        points = points_transformed_homogeneous[:, :3]

        # 提取对应彩色值并转换为RGB
        colors_bgr = color_image[valid_mask]
        colors_rgb = colors_bgr[:, ::-1] / 255.0  # BGR转RGB并归一化

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        return pcd

    def save_to_ply(self, pcd, filename):
        """保存点云为PLY格式"""
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)