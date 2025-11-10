"""
深度图像处理模块
负责将深度图像转换为3D点云的处理
"""
import numpy as np
import cv2


class DepthProcessor:
    """深度图像处理类"""

    def __init__(self):
        pass

    def apply_mask_to_depth(self, depth_image, mask):
        """将掩码应用到深度图像"""
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]

        if mask.shape != depth_image.shape:
            mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        masked_depth = depth_image * mask
        return masked_depth

    def filter_depth_by_distance(self, depth_image, min_dist_meters, max_dist_meters, depth_scale=1000.0):
        """根据距离过滤深度图像"""
        depth_in_meters = depth_image.astype(np.float32) / depth_scale
        distance_mask = (depth_in_meters >= min_dist_meters) & (depth_in_meters <= max_dist_meters)
        filtered_depth = depth_image * distance_mask.astype(depth_image.dtype)
        return filtered_depth

    def apply_object_mask_and_distance_filter(self, depth_image, mask, min_dist_meters, max_dist_meters, depth_scale=1000.0):
        """同时应用掩码和距离过滤"""
        masked_depth = self.apply_mask_to_depth(depth_image, mask)
        filtered_depth = self.filter_depth_by_distance(masked_depth, min_dist_meters, max_dist_meters, depth_scale)
        return filtered_depth

    def extract_points_in_mask(self, depth_image, mask, camera_intrinsics, depth_scale=1000.0):
        """
        从掩码区域提取3D点
        注意：坐标变换已在源头应用，此处返回变换后的点
        """
        # 应用掩码到深度图像
        masked_depth = self.apply_mask_to_depth(depth_image, mask)

        # 转换深度值为米
        depth_in_meters = masked_depth.astype(np.float32) / depth_scale

        # 获取图像尺寸
        h, w = depth_image.shape

        # 生成像素坐标
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # 转换为相机坐标
        z = depth_in_meters
        x = (u - camera_intrinsics['cx']) * z / camera_intrinsics['fx']
        y = (v - camera_intrinsics['cy']) * z / camera_intrinsics['fy']

        # 堆叠坐标
        points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # 应用掩码
        valid_mask = (z > 0) & (mask > 0)
        valid_mask = valid_mask.ravel()
        points_3d = points_3d[valid_mask]

        # 不再应用坐标变换，因为已在源头处理
        # 获取对应的深度值
        valid_depths = depth_in_meters.ravel()[valid_mask]

        return points_3d, valid_depths