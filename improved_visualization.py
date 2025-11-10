"""
改进的可视化工具 - 使用Open3D解决坐标系问题，支持彩色点云和多椭球
"""
import numpy as np
import open3d as o3d
import cv2
from ellipsoid_fitting_new import get_ellipsoid_parameters_for_grasping


class ImprovedVisualizer:
    """改进的可视化器，解决坐标系问题，支持彩色点云"""
    
    def __init__(self):
        self.vis = None
        self.geometries = []
        
    def create_ellipsoid_mesh(self, center, axes_lengths, rotation_matrix=None, color=[1, 0.7, 0]):
        """创建椭球网格"""
        # 创建单位球
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        
        # 应用缩放变换
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = axes_lengths[0]
        scale_matrix[1, 1] = axes_lengths[1]
        scale_matrix[2, 2] = axes_lengths[2]
        
        sphere.transform(scale_matrix)
        
        # 如果提供旋转矩阵，应用旋转
        if rotation_matrix is not None:
            rotation_matrix_4x4 = np.eye(4)
            rotation_matrix_4x4[:3, :3] = rotation_matrix
            sphere.transform(rotation_matrix_4x4)
        
        # 平移到中心位置
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = center
        sphere.transform(translation_matrix)
        
        # 设置颜色
        sphere.paint_uniform_color(color)
        
        return sphere

    def opencv_to_open3d_coordinates(self, point):
        """
        将OpenCV坐标系转换为Open3D坐标系
        OpenCV: X向右, Y向下, Z向前
        Open3D: X向右, Y向上, Z向前 (或不同配置)
        通常需要: Y轴翻转
        """
        # OpenCV相机坐标系到Open3D坐标系的转换
        # 一般情况下，OpenCV的Y轴向下，Open3D的Y轴向上
        return [point[0], -point[1], point[2]]

    def convert_depth_to_pointcloud_with_color(self, depth_image, color_image, camera_intrinsics, depth_scale=1000.0):
        """
        将深度图和彩色图转换为彩色点云
        解决坐标系对齐问题
        """
        # 创建Open3D相机内参
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=color_image.shape[1],
            height=color_image.shape[0],
            fx=camera_intrinsics['fx'],
            fy=camera_intrinsics['fy'],
            cx=camera_intrinsics['cx'],
            cy=camera_intrinsics['cy']
        )
        
        # 创建深度图和彩色图的几何体
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        # 创建RGBD图像
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, 
            depth_o3d,
            depth_scale=1.0/depth_scale,  # Open3D使用深度缩放的倒数
            depth_trunc=10.0,  # 最大深度截断
            convert_rgb_to_intensity=False
        )
        
        # 生成点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # OpenCV和Open3D的坐标系可能不同，需要翻转Y轴
        flip = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        pcd.transform(flip)
        
        return pcd

    def visualize_all_results(self, pcd, fitting_results, title="3D Visualization - Color Point Cloud with Fitted Ellipsoids"):
        """
        可视化所有结果：彩色点云和多个椭球
        """
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1024, height=768)
        
        # 添加点云
        vis.add_geometry(pcd)
        
        # 为每个椭球添加不同颜色
        colors = [[1, 0.7, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]  # 不同颜色
        
        for i, fitting_result in enumerate(fitting_results):
            if fitting_result and fitting_result.get('success', False):
                center = fitting_result.get('center')
                axes_lengths = fitting_result.get('axes_lengths')
                rotation_matrix = fitting_result.get('rotation_matrix')
                
                if center is not None and axes_lengths is not None:
                    # 转换坐标系
                    center_open3d = self.opencv_to_open3d_coordinates(center)
                    
                    # 创建椭球
                    ellipsoid = self.create_ellipsoid_mesh(
                        center_open3d, 
                        axes_lengths, 
                        rotation_matrix, 
                        colors[i % len(colors)]
                    )
                    
                    # 添加椭球到可视化
                    vis.add_geometry(ellipsoid)
                    
                    # 添加椭球中心点
                    center_pcd = o3d.geometry.PointCloud()
                    center_pcd.points = o3d.utility.Vector3dVector([center_open3d])
                    center_pcd.paint_uniform_color([1, 0, 0])  # 红色
                    vis.add_geometry(center_pcd)
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
        return vis


def visualize_all_fitting_results(color_image, depth_image, camera_intrinsics, masks, fitting_results_list):
    """
    可视化所有拟合结果的主函数
    """
    visualizer = ImprovedVisualizer()
    
    # 转换为彩色点云
    pcd = visualizer.convert_depth_to_pointcloud_with_color(
        depth_image, 
        color_image, 
        camera_intrinsics
    )
    
    # 可视化所有结果
    visualizer.visualize_all_results(pcd, fitting_results_list)
    
    return pcd


def coordinate_opencv_to_open3d(points):
    """
    转换OpenCV坐标系到Open3D坐标系
    """
    if len(points.shape) == 2:
        # 多个点 (N, 3)
        transformed_points = points.copy()
        transformed_points[:, 1] = -transformed_points[:, 1]  # Y轴翻转
        return transformed_points
    else:
        # 单个点 (3,)
        return [points[0], -points[1], points[2]]