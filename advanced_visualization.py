"""
高级Open3D可视化模块 - 显示彩色点云和椭球拟合结果
"""
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
import math


class AdvancedVisualizer:
    """高级可视化器，专门优化彩色点云显示和椭球拟合结果"""
    
    def __init__(self):
        self.vis = None
        
    def create_colored_point_cloud(self, depth_image, color_image, camera_intrinsics, depth_scale=1000.0, mask=None):
        """
        创建彩色点云
        """
        # 获取相机内参
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics["fy"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]
        
        # 获取图像尺寸
        h, w = depth_image.shape
        
        # 生成像素坐标
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 将像素坐标转换为相机坐标
        z = depth_image / depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # 堆叠坐标
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # 应用掩码（如果提供）
        if mask is not None:
            valid_mask = (z > 0) & (mask > 0)  # 只保留有效深度和掩码区域
            valid_mask = valid_mask.ravel()
        else:
            valid_mask = z.ravel() > 0  # 只保留有效深度
        
        points = points[valid_mask]
        
        # 获取对应的颜色
        color_bgr = cv2.resize(color_image, (w, h))
        colors = color_bgr.reshape(-1, 3)[valid_mask]
        
        # 归一化颜色到[0,1]
        colors = colors.astype(np.float64) / 255.0
        
        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def create_ellipsoid_wireframe(self, center, axes_lengths, rotation_matrix=None, color=[1, 0.7, 0], line_width=2.0):
        """
        创建椭球的线框骨架，避免遮挡点云
        """
        # 创建参数化的椭球表面点
        u = np.linspace(0, 2 * np.pi, 30)  # 经度（减少点数以提高性能）
        v = np.linspace(0, np.pi, 15)      # 纬度（减少点数以提高性能）
        
        all_points = []
        all_lines = []
        line_colors = []
        
        # 经线（纵向线）
        for i in range(len(u)):
            longitude_points = []
            for j in range(len(v)):
                # 标准椭球参数方程
                x = axes_lengths[0] * np.cos(u[i]) * np.sin(v[j])
                y = axes_lengths[1] * np.sin(u[i]) * np.sin(v[j])
                z = axes_lengths[2] * np.cos(v[j])
                longitude_points.append([x, y, z])
            
            # 转换到全局坐标系
            longitude_points = np.array(longitude_points)
            if rotation_matrix is not None:
                longitude_points = longitude_points @ rotation_matrix.T
            longitude_points += center
            
            all_points.extend(longitude_points)
            
            # 创建线段
            for k in range(len(longitude_points)-1):
                start_idx = len(all_points) - len(longitude_points) + k
                end_idx = start_idx + 1
                all_lines.append([start_idx, end_idx])
                line_colors.append(color)
        
        # 纬线（横向线）
        for j in range(len(v)):
            latitude_points = []
            for i in range(len(u)):
                # 标准椭球参数方程
                x = axes_lengths[0] * np.cos(u[i]) * np.sin(v[j])
                y = axes_lengths[1] * np.sin(u[i]) * np.sin(v[j])
                z = axes_lengths[2] * np.cos(v[j])
                latitude_points.append([x, y, z])
            
            # 转换到全局坐标系
            latitude_points = np.array(latitude_points)
            if rotation_matrix is not None:
                latitude_points = latitude_points @ rotation_matrix.T
            latitude_points += center
            
            all_points.extend(latitude_points)
            
            # 创建线段（连接纬线上的点）
            for k in range(len(latitude_points)-1):
                start_idx = len(all_points) - len(latitude_points) + k
                end_idx = start_idx + 1
                all_lines.append([start_idx, end_idx])
                line_colors.append(color)
            
            # 连接最后一个点到第一个点形成环
            start_idx = len(all_points) - len(latitude_points)
            end_idx = len(all_points) - 1
            all_lines.append([end_idx, start_idx])  # 连接最后一个点到第一个点
            line_colors.append(color)
        
        # 创建LineSet
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(all_points)
        lineset.lines = o3d.utility.Vector2iVector(all_lines)
        lineset.colors = o3d.utility.Vector3dVector(line_colors)
        
        return lineset

    def create_ellipsoid_principal_axes(self, center, axes_lengths, rotation_matrix=None, color=[1, 0, 0]):
        """
        创建椭球的主轴，清晰显示三个轴的长度和方向
        """
        # 定义三个主轴方向（考虑旋转）
        if rotation_matrix is not None:
            axes_dirs = rotation_matrix.T  # 旋转矩阵的列向量是主轴方向
        else:
            axes_dirs = np.eye(3)  # 单位矩阵，标准轴方向
        
        # 为每个轴创建线段
        all_points = []
        all_lines = []
        line_colors = []
        
        for i in range(3):
            axis_dir = axes_dirs[:, i]
            axis_length = axes_lengths[i]
            
            # 轴的起点和终点
            start_point = center - axis_dir * axis_length
            end_point = center + axis_dir * axis_length
            
            # 添加轴线的两个端点
            point_start_idx = len(all_points)
            all_points.append(start_point)
            all_points.append(end_point)
            
            # 创建轴线
            all_lines.append([point_start_idx, point_start_idx + 1])
            line_colors.append(color)
        
        # 创建LineSet
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(all_points)
        lineset.lines = o3d.utility.Vector2iVector(all_lines)
        lineset.colors = o3d.utility.Vector3dVector(line_colors)
        
        return lineset

    def create_coordinate_frame(self, center, scale=0.1):
        """
        创建坐标系框架，更好地显示方向
        """
        # 创建坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale, origin=center
        )
        return coord_frame

    def visualize_fitting_results_with_colored_pointcloud(
        self, 
        depth_image, 
        color_image, 
        camera_intrinsics, 
        fitting_results_list, 
        masks_list=None,
        title="3D Visualization - Color Point Cloud with Fitted Ellipsoids"
    ):
        """
        可视化彩色点云和椭球拟合结果
        """
        print(f"创建彩色点云...")
        
        # 创建完整的彩色点云（不应用掩码，显示全部场景）
        full_point_cloud = self.create_colored_point_cloud(
            depth_image, 
            color_image, 
            camera_intrinsics
        )
        
        print(f"点云创建完成，共 {len(full_point_cloud.points)} 个点")
        
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=900)
        
        # 添加完整的点云
        vis.add_geometry(full_point_cloud)
        
        print(f"添加了 {len(fitting_results_list)} 个椭球拟合结果")
        
        # 为每个椭球添加不同颜色和坐标系
        colors = [
            [1, 0.7, 0],     # 橙色
            [0, 1, 0],       # 绿色
            [0, 0.7, 1],     # 蓝色
            [1, 0, 1],       # 紫色
            [0, 1, 1],       # 青色
            [1, 1, 0],       # 黄色
            [1, 0.5, 0.5],   # 粉色
        ]
        
        for i, fitting_result in enumerate(fitting_results_list):
            if fitting_result and fitting_result.get('success', False):
                center = fitting_result.get('center')
                axes_lengths = fitting_result.get('axes_lengths')
                rotation_matrix = fitting_result.get('rotation_matrix')
                
                if center is not None and axes_lengths is not None:
                    # 创建椭球
                    ellipsoid = self.create_ellipsoid_mesh(
                        center=center,
                        axes_lengths=axes_lengths,
                        rotation_matrix=rotation_matrix,
                        color=colors[i % len(colors)],
                        transparency=0.3  # 半透明
                    )
                    
                    # 添加椭球到可视化
                    vis.add_geometry(ellipsoid)
                    
                    # 添加椭球中心点（红色大点）
                    center_pcd = o3d.geometry.PointCloud()
                    center_pcd.points = o3d.utility.Vector3dVector([center])
                    center_pcd.paint_uniform_color([1, 0, 0])  # 红色
                    # 通过添加多个点来创建更大的视觉效果
                    vis.add_geometry(center_pcd)
                    
                    # 添加坐标系框架
                    coord_frame = self.create_coordinate_frame(center, scale=max(axes_lengths) * 0.3)
                    vis.add_geometry(coord_frame)
                    
                    print(f"  - 椭球 {i+1}: 中心 {center}, 轴长 {axes_lengths}")
        
        # 设置视角
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
        
        # 运行可视化
        print("启动可视化窗口...")
        print("操作说明：")
        print("  - 鼠标左键：旋转视角")
        print("  - 鼠标右键：平移")
        print("  - 滚轮：缩放")
        print("  - 按 'q' 或关闭窗口退出")
        
        vis.run()
        vis.destroy_window()
        
        print("可视化完成")
        return vis

    def visualize_single_object_with_ellipsoid(
        self,
        depth_image,
        color_image, 
        camera_intrinsics,
        fitting_result,
        mask=None,
        title="Single Object - Color Point Cloud with Fitted Ellipsoid"
    ):
        """
        可视化单个物体的彩色点云和椭球拟合结果
        """
        print("创建单个物体的彩色点云...")
        
        # 创建带掩码的彩色点云
        object_point_cloud = self.create_colored_point_cloud(
            depth_image, 
            color_image, 
            camera_intrinsics,
            mask=mask
        )
        
        print(f"物体点云创建完成，共 {len(object_point_cloud.points)} 个点")
        
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=900)
        
        # 添加物体点云
        vis.add_geometry(object_point_cloud)
        
        # 如果有拟合结果，添加椭球
        if fitting_result and fitting_result.get('success', False):
            center = fitting_result.get('center')
            axes_lengths = fitting_result.get('axes_lengths')
            rotation_matrix = fitting_result.get('rotation_matrix')
            
            if center is not None and axes_lengths is not None:
                # 创建椭球（高对比度颜色）
                ellipsoid = self.create_ellipsoid_mesh(
                    center=center,
                    axes_lengths=axes_lengths,
                    rotation_matrix=rotation_matrix,
                    color=[1, 0.2, 0.2],  # 鲜艳的红色
                    transparency=0.4
                )
                
                vis.add_geometry(ellipsoid)
                
                # 添加椭球中心（大红点）
                center_pcd = o3d.geometry.PointCloud()
                center_pcd.points = o3d.utility.Vector3dVector([center])
                center_pcd.paint_uniform_color([1, 0, 0])
                vis.add_geometry(center_pcd)
                
                # 添加坐标系
                coord_frame = self.create_coordinate_frame(center, scale=max(axes_lengths) * 0.5)
                vis.add_geometry(coord_frame)
                
                print(f"椭球拟合结果：中心 {center}, 轴长 {axes_lengths}")
        
        # 设置渲染选项
        vis.get_render_option().point_size = 3.0  # 更大的点大小
        vis.get_render_option().background_color = np.array([0.95, 0.95, 0.95])  # 浅灰色背景，更好显示彩色点云
        
        print("启动单物体可视化窗口...")
        print("操作说明：")
        print("  - 鼠标左键：旋转视角")
        print("  - 鼠标右键：平移")
        print("  - 滚轮：缩放")
        print("  - 按 'q' 或关闭窗口退出")
        
        vis.run()
        vis.destroy_window()
        
        print("单物体可视化完成")
        return vis


def visualize_all_objects_with_colored_pointcloud(
    depth_image, 
    color_image, 
    camera_intrinsics, 
    fitting_results_list, 
    masks_list=None
):
    """
    可视化所有物体的彩色点云和椭球拟合结果
    """
    visualizer = AdvancedVisualizer()
    return visualizer.visualize_fitting_results_with_colored_pointcloud(
        depth_image, 
        color_image, 
        camera_intrinsics, 
        fitting_results_list, 
        masks_list
    )


def visualize_single_object_with_colored_pointcloud(
    depth_image,
    color_image, 
    camera_intrinsics,
    fitting_result,
    mask=None
):
    """
    可视化单个物体的彩色点云和椭球拟合结果
    """
    visualizer = AdvancedVisualizer()
    return visualizer.visualize_single_object_with_ellipsoid(
        depth_image,
        color_image, 
        camera_intrinsics,
        fitting_result,
        mask
    )