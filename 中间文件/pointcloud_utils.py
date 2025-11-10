import numpy as np
import open3d as o3d
import matplotlib.cm  # 用于颜色映射


class DepthToPointCloudConverter:
    """
    一个用于将深度图转换为 3D 点云的类。

    这个转换依赖于相机的内参矩阵。
    现在它还支持基于距离的过滤和着色。
    """

    def __init__(self, fx, fy, cx, cy, depth_scale=1000.0):
        """
        初始化转换器。

        参数:
        - fx: x 轴的焦距 (单位: 像素)
        - fy: y 轴的焦距 (单位: 像素)
        - cx: x 轴的主点 (光学中心, 单位: 像素)
        - cy: y 轴的主点 (光学中心, 单位: 像素)
        - depth_scale: 将深度图中的原始值转换
                       为米 (meters) 的比例因子。
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        print(f"点云转换器已初始化 (fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy})")

    def convert(self,
                depth_map: np.ndarray,
                min_dist: float = 0.55,
                max_dist: float = 8.0,
                colormap_name: str = 'jet') -> o3d.geometry.PointCloud:
        """
        将单帧深度图转换为一个着色的 Open3D 点云对象。

        参数:
        - depth_map: (H, W) 的 NumPy 数组 (原始深度值, uint16)。
        - min_dist:  (米) 要保留的最小距离。
        - max_dist:  (米) 要保留的最大距离 (也用于颜色归一化)。
        - colormap_name: 用于着色的 matplotlib 调色板名称。

        返回:
        - pcd: 一个 Open3D PointCloud 对象，包含点和颜色。
        """

        # 1. 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()

        # 2. 将原始深度值转换为米 (float32)
        height, width = depth_map.shape
        z_meters = depth_map.astype(np.float32) / self.depth_scale

        # 3. !! 过滤 !!
        #    创建有效掩码 (mask)
        #    我们只关心在 [min_dist, max_dist] 范围内的点
        valid_mask = (z_meters >= min_dist) & (z_meters <= max_dist)

        # 4. 创建像素坐标网格 (v, u)
        v, u = np.indices((height, width))

        # 5. 提取有效点的 (z, u, v)
        z_valid = z_meters[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        if len(z_valid) == 0:
            print("警告: 过滤后没有剩余的点。")
            return pcd  # 返回一个空点云

        # 6. 从 (u, v, z) 计算 (x, y, z)
        #    x = (u - cx) * z / fx
        #    y = (v - cy) * z / fy
        x_valid = (u_valid - self.cx) * z_valid / self.fx
        y_valid = (v_valid - self.cy) * z_valid / self.fy

        # 组合为 (N, 3) 的点
        points = np.stack((x_valid, y_valid, z_valid), axis=-1)

        # 7. !! 上色 !!
        #    根据 z 值 (距离) 对点进行着色

        # 7.1 将距离归一化到 [0, 1] 范围
        #     (z_valid - min_dist) / (max_dist - min_dist)
        norm_z = (z_valid - min_dist) / (max_dist - min_dist)

        # 7.2 获取 matplotlib 调色板
        cmap = matplotlib.cm.get_cmap(colormap_name)

        # 7.3 将归一化的 z 值映射为 (N, 4) RGBA 颜色
        #     我们只取前 3 个通道 (RGB)
        colors_rgb = cmap(norm_z)[:, :3]

        # 8. 将点和颜色添加到 Open3D 对象
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        return pcd

    def convert_with_color(self, 
                           depth_map: np.ndarray, 
                           color_image: np.ndarray, 
                           min_dist: float = 0.55, 
                           max_dist: float = 8.0) -> o3d.geometry.PointCloud:
        """
        将深度图和彩色图转换为彩色点云

        参数:
        - depth_map: (H, W) 的 NumPy 数组 (原始深度值, uint16)
        - color_image: (H, W, 3) 的彩色图像
        - min_dist: (米) 要保留的最小距离
        - max_dist: (米) 要保留的最大距离

        返回:
        - pcd: 一个 Open3D PointCloud 对象，包含点和颜色
        """
        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()

        # 将原始深度值转换为米 (float32)
        height, width = depth_map.shape
        z_meters = depth_map.astype(np.float32) / self.depth_scale

        # 过滤深度值
        valid_mask = (z_meters >= min_dist) & (z_meters <= max_dist)

        # 创建像素坐标网格
        v, u = np.indices((height, width))

        # 提取有效点的 (z, u, v)
        z_valid = z_meters[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        if len(z_valid) == 0:
            print("警告: 过滤后没有剩余的点。")
            return pcd  # 返回一个空点云

        # 从 (u, v, z) 计算 (x, y, z)
        x_valid = (u_valid - self.cx) * z_valid / self.fx
        y_valid = (v_valid - self.cy) * z_valid / self.fy

        # 组合为 (N, 3) 的点
        points = np.stack((x_valid, y_valid, z_valid), axis=-1)

        # 提取对应像素的彩色值
        colors_bgr = color_image[valid_mask]
        # 转换BGR到RGB并将值归一化到[0,1]
        colors_rgb = colors_bgr[:, ::-1] / 255.0  # BGR转RGB，并归一化

        # 将点和颜色添加到 Open3D 对象
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        return pcd

    def save_to_ply(self, pcd: o3d.geometry.PointCloud, filename: str):
        """
        将 Open3D 点云对象保存为 .ply 文件。
        Open3D 会自动处理带有颜色的头部和数据。

        参数:
        - pcd: Open3D PointCloud 对象。
        - filename: 输出的 .ply 文件路径。
        """
        try:
            # write_ascii=True 使 .ply 文件易于阅读
            o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
            print(f"成功: {len(pcd.points)} 个点 (已着色并过滤) 保存到 {filename}")
        except Exception as e:
            print(f"错误: 无法保存 PLY 文件到 {filename}. 错误信息: {e}")