"""
可视化工具模块：用于三维点云和椭球拟合结果的可视化
解决OpenCV和Open3D坐标系不一致的问题
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import cv2
import math


class Arrow3D(FancyArrowPatch):
    """3D箭头类，用于绘制坐标轴"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
    def do_3d_projection(self, renderer=None):
        # 添加此方法以解决matplotlib 3D投影问题
        return 0


class PointCloudVisualizer:
    """点云和椭球可视化器"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def create_coordinate_frame(self, origin, x_axis, y_axis, z_axis, scale=0.1):
        """创建坐标系框架"""
        # 绘制X轴 (红色)
        x_arrow = Arrow3D(
            [origin[0], origin[0] + x_axis[0] * scale],
            [origin[1], origin[1] + x_axis[1] * scale],
            [origin[2], origin[2] + x_axis[2] * scale],
            mutation_scale=10, arrowstyle="-|>", color="r", label="X"
        )
        
        # 绘制Y轴 (绿色)
        y_arrow = Arrow3D(
            [origin[0], origin[0] + y_axis[0] * scale],
            [origin[1], origin[1] + y_axis[1] * scale],
            [origin[2], origin[2] + y_axis[2] * scale],
            mutation_scale=10, arrowstyle="-|>", color="g", label="Y"
        )
        
        # 绘制Z轴 (蓝色)
        z_arrow = Arrow3D(
            [origin[0], origin[0] + z_axis[0] * scale],
            [origin[1], origin[1] + z_axis[1] * scale],
            [origin[2], origin[2] + z_axis[2] * scale],
            mutation_scale=10, arrowstyle="-|>", color="b", label="Z"
        )
        
        return [x_arrow, y_arrow, z_arrow]
    
    def draw_ellipsoid(self, center, axes_lengths, rotation_matrix=None, n_points=100):
        """绘制椭球体"""
        # 创建参数网格
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        
        # 应用轴长缩放
        x = x * axes_lengths[0]
        y = y * axes_lengths[1]
        z = z * axes_lengths[2]
        
        # 应用旋转矩阵（如果提供）
        if rotation_matrix is not None:
            # 将网格点重塑为列向量
            points = np.vstack((x.ravel(), y.ravel(), z.ravel()))
            # 应用旋转
            rotated_points = rotation_matrix @ points
            # 重新塑造为网格
            x = rotated_points[0, :].reshape(x.shape)
            y = rotated_points[1, :].reshape(y.shape)
            z = rotated_points[2, :].reshape(z.shape)
        
        # 平移到中心位置
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        return x, y, z
    
    def visualize_fitting_result(self, points_3d, fitting_result, title="Ellipsoid Fitting Result"):
        """
        可视化椭球拟合结果
        points_3d: 3D点云
        fitting_result: 椭球拟合结果字典
        """
        if self.fig is not None:
            plt.close(self.fig)
            
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        if len(points_3d) > 0:
            # 如果点云有颜色信息（RGB值），使用它们
            if points_3d.shape[1] >= 6:  # 有RGB信息 (x,y,z,r,g,b)
                colors = points_3d[:, 3:6] / 255.0  # 归一化到[0,1]
                scatter = self.ax.scatter(
                    points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                    c=colors, s=1, alpha=0.7, label='Point Cloud'
                )
            else:  # 只有XYZ坐标
                scatter = self.ax.scatter(
                    points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                    c='gray', s=1, alpha=0.7, label='Point Cloud'
                )
        
        # 如果拟合成功，绘制椭球
        if fitting_result and fitting_result.get('success', False):
            center = fitting_result['center']
            axes_lengths = fitting_result['axes_lengths']
            rotation_matrix = fitting_result.get('rotation_matrix', None)
            
            if center is not None and axes_lengths is not None:
                # 绘制椭球
                x, y, z = self.draw_ellipsoid(center, axes_lengths, rotation_matrix)
                self.ax.plot_surface(x, y, z, alpha=0.3, color='yellow', label='Fitted Ellipsoid')
                
                # 标记椭球中心
                self.ax.scatter(*center, color='red', s=100, marker='x', label='Ellipsoid Center')
                
                # 显示椭球参数
                self.ax.text(center[0], center[1], center[2] + max(axes_lengths) + 0.02,
                            f'Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n'
                            f'Axes: ({axes_lengths[0]:.3f}, {axes_lengths[1]:.3f}, {axes_lengths[2]:.3f})',
                            fontsize=10)
        
        # 添加坐标轴
        origin = [0, 0, 0]
        x_axis = [1, 0, 0]
        y_axis = [0, 1, 0]
        z_axis = [0, 0, 1]
        
        arrows = self.create_coordinate_frame(origin, x_axis, y_axis, z_axis, scale=0.3)
        for arrow in arrows:
            self.ax.add_artist(arrow)
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(title)
        
        # 自动调整视角以适应数据范围
        if len(points_3d) > 0:
            ranges = np.max(points_3d[:, :3], axis=0) - np.min(points_3d[:, :3], axis=0)
            max_range = np.max(ranges)
            mean_pos = np.mean(points_3d[:, :3], axis=0)
            
            self.ax.set_xlim(mean_pos[0] - max_range/2, mean_pos[0] + max_range/2)
            self.ax.set_ylim(mean_pos[1] - max_range/2, mean_pos[1] + max_range/2)
            self.ax.set_zlim(mean_pos[2] - max_range/2, mean_pos[2] + max_range/2)
        
        # 添加图例
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return self.fig, self.ax
    
    def visualize_robot_grasp_pose(self, fitting_result, gripper_offset=0.1):
        """
        为机器人抓取可视化姿态
        fitting_result: 椭球拟合结果
        gripper_offset: 机器人夹爪距离物体表面的偏移量
        """
        if not (fitting_result and fitting_result.get('success', False)):
            print("无法可视化机器人抓取姿态：椭球拟合失败")
            return None
            
        center = fitting_result['center']
        axes_lengths = fitting_result['axes_lengths']
        rotation_matrix = fitting_result.get('rotation_matrix', np.eye(3))
        
        if center is None or axes_lengths is None:
            print("无法可视化机器人抓取姿态：椭球参数缺失")
            return None
        
        # 计算主要轴（最长轴），这通常是最适合抓取的方向
        major_axis_idx = np.argmax(axes_lengths)
        major_axis = rotation_matrix[:, major_axis_idx]  # 主轴方向
        
        # 计算抓取位置（在主轴方向的物体表面上）
        grasp_distance = axes_lengths[major_axis_idx] / 2 + gripper_offset
        grasp_position = center + major_axis * grasp_distance
        
        # 计算抓取方向（垂直于主轴）
        # 这里简化为沿着主轴方向抓取
        approach_vector = major_axis
        binormal_vector = rotation_matrix[:, (major_axis_idx + 1) % 3]
        orientation_matrix = np.column_stack([approach_vector, binormal_vector, 
                                             np.cross(approach_vector, binormal_vector)])
        
        # 可视化抓取姿态
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 9))
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绘制抓取方向箭头
        arrow_length = 0.05
        grasp_arrow = Arrow3D(
            [grasp_position[0], grasp_position[0] + approach_vector[0] * arrow_length],
            [grasp_position[1], grasp_position[1] + approach_vector[1] * arrow_length],
            [grasp_position[2], grasp_position[2] + approach_vector[2] * arrow_length],
            mutation_scale=20, arrowstyle="-|>", color="magenta", label="Grasp Direction"
        )
        self.ax.add_artist(grasp_arrow)
        
        # 绘制抓取点
        self.ax.scatter(*grasp_position, color='magenta', s=100, marker='o', 
                       label='Grasp Point')
        
        # 添加抓取信息文本
        self.ax.text(grasp_position[0], grasp_position[1], grasp_position[2] + 0.05,
                    f'Grasp Pose:\nPosition: ({grasp_position[0]:.3f}, {grasp_position[1]:.3f}, {grasp_position[2]:.3f})\n'
                    f'Approach: ({approach_vector[0]:.3f}, {approach_vector[1]:.3f}, {approach_vector[2]:.3f})',
                    fontsize=10)
        
        return {
            'grasp_position': grasp_position,
            'approach_vector': approach_vector,
            'orientation_matrix': orientation_matrix,
            'major_axis': major_axis,
            'axes_lengths': axes_lengths
        }


def plot_ellipsoid_fitting_result(points_3d, fitting_result):
    """
    可视化椭球拟合结果的便捷函数
    """
    visualizer = PointCloudVisualizer()
    return visualizer.visualize_fitting_result(points_3d, fitting_result)


def get_ellipsoid_parameters_for_grasping(fitting_result):
    """
    为机器人抓取提取椭球参数
    返回适合机器人抓取计算的数据
    """
    if not (fitting_result and fitting_result.get('success', False)):
        return None
        
    center = fitting_result.get('center')
    axes_lengths = fitting_result.get('axes_lengths')
    rotation_matrix = fitting_result.get('rotation_matrix', np.eye(3))
    
    if center is None or axes_lengths is None:
        return None
    
    # 返回物体的基本物理参数
    params = {
        'center': center,                    # 物体中心
        'axes_lengths': axes_lengths,        # 三轴长度
        'rotation_matrix': rotation_matrix,  # 旋转矩阵
        'volume': (4/3) * math.pi * axes_lengths[0] * axes_lengths[1] * axes_lengths[2],  # 体积估算
        'mass': None,                        # 质量（需要根据材料估算）
        'principal_axis': rotation_matrix[:, np.argmax(axes_lengths)],  # 主轴方向
        'size_category': 'small' if max(axes_lengths) < 0.1 else 'medium' if max(axes_lengths) < 0.2 else 'large'  # 尺寸分类
    }
    
    return params


def coordinate_opencv_to_world(depth_image, color_image, u, v, depth_value, camera_intrinsics):
    """
    将OpenCV图像坐标(u,v)和深度值转换为世界坐标系
    这解决了坐标系不一致的问题
    """
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    depth_scale = 1000.0  # 假设深度值以毫米为单位
    
    # 将像素坐标转换为相机坐标系
    z = depth_value / depth_scale  # 转换为米
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # OpenCV相机坐标系: X向右, Y向下, Z向前
    # 转换为标准世界坐标系: X向右, Y向上, Z向前 (或根据需要调整)
    # 注意：根据您的具体应用，可能需要额外的坐标变换
    world_point = np.array([x, -y, z])  # Y轴翻转以匹配某些世界坐标系统
    
    return world_point


if __name__ == "__main__":
    # 测试代码
    # 生成一些测试数据
    t = np.linspace(0, 2*np.pi, 100)
    s = np.linspace(0, np.pi, 100)
    t, s = np.meshgrid(t, s)
    
    # 创建一个椭球形状的点云
    a, b, c = 0.1, 0.05, 0.08  # 椭球的三个半轴
    x = a * np.cos(t) * np.sin(s)
    y = b * np.sin(t) * np.sin(s)
    z = c * np.cos(s)
    
    # 添加一些噪声
    x += np.random.normal(0, 0.005, x.shape)
    y += np.random.normal(0, 0.005, y.shape)
    z += np.random.normal(0, 0.005, z.shape)
    
    # 将椭球展平为点云
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    center = np.mean(points, axis=0)
    
    # 创建拟合结果模拟数据
    fitting_result = {
        'success': True,
        'center': center,
        'axes_lengths': np.array([a, b, c]),
        'type': 'ellipsoid',
        'num_inliers': len(points),
        'rotation_matrix': np.eye(3)  # 单位矩阵表示没有旋转
    }
    
    # 可视化
    visualizer = PointCloudVisualizer()
    visualizer.visualize_fitting_result(points, fitting_result)