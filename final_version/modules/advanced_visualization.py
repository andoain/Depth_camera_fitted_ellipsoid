"""
高级可视化模块
用于显示彩色点云和椭球拟合结果
"""
import numpy as np
import open3d as o3d
import cv2


class AdvancedVisualizer:
    """高级可视化器"""
    
    def __init__(self):
        self.vis = None

    def create_colored_point_cloud(self, depth_image, color_image, camera_intrinsics, depth_scale=1000.0, mask=None):
        """创建彩色点云"""
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics["fy"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]
        
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        z = depth_image / depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        if mask is not None:
            valid_mask = (z > 0) & (mask > 0)
            valid_mask = valid_mask.ravel()
        else:
            valid_mask = z.ravel() > 0
        
        points = points[valid_mask]
        
        color_bgr = cv2.resize(color_image, (w, h))
        colors = color_bgr.reshape(-1, 3)[valid_mask]
        colors = colors.astype(np.float64) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def create_ellipsoid_wireframe(self, center, axes_lengths, rotation_matrix=None, color=[1, 0.7, 0], line_width=2.0):
        """创建椭球的线框模型"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        
        all_points = []
        all_lines = []
        line_colors = []

        # 经线（纵向线）
        for i in range(len(u)):
            longitude_points = []
            for j in range(len(v)):
                x = axes_lengths[0] * np.cos(u[i]) * np.sin(v[j])
                y = axes_lengths[1] * np.sin(u[i]) * np.sin(v[j])
                z = axes_lengths[2] * np.cos(v[j])
                longitude_points.append([x, y, z])

            longitude_points = np.array(longitude_points)
            if rotation_matrix is not None:
                longitude_points = longitude_points @ rotation_matrix.T
            longitude_points += center

            all_points.extend(longitude_points)
            
            for k in range(len(longitude_points)-1):
                start_idx = len(all_points) - len(longitude_points) + k
                end_idx = start_idx + 1
                all_lines.append([start_idx, end_idx])
                line_colors.append(color)

        # 纬线（横向线）
        for j in range(len(v)):
            latitude_points = []
            for i in range(len(u)):
                x = axes_lengths[0] * np.cos(u[i]) * np.sin(v[j])
                y = axes_lengths[1] * np.sin(u[i]) * np.sin(v[j])
                z = axes_lengths[2] * np.cos(v[j])
                latitude_points.append([x, y, z])

            latitude_points = np.array(latitude_points)
            if rotation_matrix is not None:
                latitude_points = latitude_points @ rotation_matrix.T
            latitude_points += center

            all_points.extend(latitude_points)
            
            for k in range(len(latitude_points)-1):
                start_idx = len(all_points) - len(latitude_points) + k
                end_idx = start_idx + 1
                all_lines.append([start_idx, end_idx])
                line_colors.append(color)
            
            start_idx = len(all_points) - len(latitude_points)
            end_idx = len(all_points) - 1
            all_lines.append([end_idx, start_idx])
            line_colors.append(color)

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(all_points)
        lineset.lines = o3d.utility.Vector2iVector(all_lines)
        lineset.colors = o3d.utility.Vector3dVector(line_colors)
        
        return lineset

    def create_ellipsoid_principal_axes(self, center, axes_lengths, rotation_matrix=None, color=[1, 0, 0]):
        """创建椭球的主轴"""
        if rotation_matrix is not None:
            axes_dirs = rotation_matrix.T
        else:
            axes_dirs = np.eye(3)
        
        all_points = []
        all_lines = []
        line_colors = []
        
        for i in range(3):
            axis_dir = axes_dirs[:, i]
            axis_length = axes_lengths[i]
            
            start_point = center - axis_dir * axis_length
            end_point = center + axis_dir * axis_length
            
            point_start_idx = len(all_points)
            all_points.append(start_point)
            all_points.append(end_point)
            
            all_lines.append([point_start_idx, point_start_idx + 1])
            line_colors.append(color)

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(all_points)
        lineset.lines = o3d.utility.Vector2iVector(all_lines)
        lineset.colors = o3d.utility.Vector3dVector(line_colors)
        
        return lineset

    def visualize_fitting_results_with_colored_pointcloud(
        self, 
        depth_image, 
        color_image, 
        camera_intrinsics, 
        fitting_results_list, 
        masks_list=None,
        title="3D Visualization - Color Point Cloud with Fitted Ellipsoids (Wireframe)"
    ):
        """可视化点云和椭球拟合结果"""
        print(f"创建彩色点云...")
        
        full_point_cloud = self.create_colored_point_cloud(
            depth_image, 
            color_image, 
            camera_intrinsics
        )
        
        print(f"点云创建完成，共 {len(full_point_cloud.points)} 个点")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=900)
        
        # 添加点云
        vis.add_geometry(full_point_cloud)
        
        print(f"添加了 {len(fitting_results_list)} 个椭球拟合结果")
        
        colors = [
            [1, 0.7, 0], [0, 1, 0], [0, 0.7, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0.5, 0.5]
        ]

        for i, fitting_result in enumerate(fitting_results_list):
            if fitting_result and fitting_result.get('success', False):
                center = fitting_result.get('center')
                axes_lengths = fitting_result.get('axes_lengths')
                rotation_matrix = fitting_result.get('rotation_matrix')
                
                if center is not None and axes_lengths is not None:
                    # 创建椭球线框
                    ellipsoid_wireframe = self.create_ellipsoid_wireframe(
                        center=center,
                        axes_lengths=axes_lengths,
                        rotation_matrix=rotation_matrix,
                        color=colors[i % len(colors)],
                        line_width=2.0
                    )

                    vis.add_geometry(ellipsoid_wireframe)
                    
                    # 添加主轴
                    principal_axes = self.create_ellipsoid_principal_axes(
                        center=center,
                        axes_lengths=axes_lengths,
                        rotation_matrix=rotation_matrix,
                        color=[1, 0, 0]
                    )
                    
                    vis.add_geometry(principal_axes)
                    
                    # 添加椭球中心点
                    center_pcd = o3d.geometry.PointCloud()
                    center_pcd.points = o3d.utility.Vector3dVector([center])
                    center_pcd.paint_uniform_color([1, 0, 0])
                    vis.add_geometry(center_pcd)
                    
                    print(f"  - 椭球 {i+1}: 中心 {center}, 轴长 {axes_lengths}")

        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
        
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


def visualize_all_objects_with_colored_pointcloud(
    depth_image, 
    color_image, 
    camera_intrinsics, 
    fitting_results_list, 
    masks_list=None
):
    """可视化所有物体的拟合结果"""
    visualizer = AdvancedVisualizer()
    return visualizer.visualize_fitting_results_with_colored_pointcloud(
        depth_image, 
        color_image, 
        camera_intrinsics, 
        fitting_results_list, 
        masks_list
    )