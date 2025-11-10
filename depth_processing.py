"""
深度图像处理模块
用于处理深度图像，结合YOLO检测结果保留目标区域的深度数据
"""
import numpy as np
import cv2


class DepthProcessor:
    """
    深度图像处理器
    用于应用YOLO检测掩码到深度图像上，提取目标区域的深度数据
    """
    def __init__(self):
        pass
    
    def apply_mask_to_depth(self, depth_image, mask):
        """
        将掩码应用到深度图像上
        
        Args:
            depth_image: 深度图像 (H, W) 或 (H, W, C)
            mask: 掩码图像，1表示目标区域，0表示背景 (H, W)
            
        Returns:
            masked_depth: 应用掩码后的深度图像
        """
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        
        # 确保掩码和深度图尺寸一致
        if mask.shape != depth_image.shape:
            mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 应用掩码
        masked_depth = depth_image * mask
        
        return masked_depth
    
    def filter_depth_by_distance(self, depth_image, min_dist_meters, max_dist_meters, depth_scale=1000.0):
        """
        根据距离范围过滤深度图像
        
        Args:
            depth_image: 深度图像
            min_dist_meters: 最小距离（米）
            max_dist_meters: 最大距离（米）
            depth_scale: 深度缩放因子，用于将原始值转换为米
            
        Returns:
            filtered_depth: 过滤后的深度图像
        """
        # 将深度值转换为米
        depth_in_meters = depth_image.astype(np.float32) / depth_scale
        
        # 创建距离掩码
        distance_mask = (depth_in_meters >= min_dist_meters) & (depth_in_meters <= max_dist_meters)
        
        # 应用距离掩码
        filtered_depth = depth_image * distance_mask.astype(depth_image.dtype)
        
        return filtered_depth
    
    def apply_object_mask_and_distance_filter(self, depth_image, mask, min_dist_meters, max_dist_meters, depth_scale=1000.0):
        """
        同时应用目标掩码和距离过滤
        
        Args:
            depth_image: 深度图像
            mask: 目标掩码
            min_dist_meters: 最小距离（米）
            max_dist_meters: 最大距离（米）
            depth_scale: 深度缩放因子
            
        Returns:
            filtered_depth: 同时应用了目标掩码和距离过滤的深度图像
        """
        # 首先应用目标掩码
        masked_depth = self.apply_mask_to_depth(depth_image, mask)
        
        # 然后应用距离过滤
        filtered_depth = self.filter_depth_by_distance(masked_depth, min_dist_meters, max_dist_meters, depth_scale)
        
        return filtered_depth
    
    def extract_points_in_mask(self, depth_image, mask, camera_intrinsics, depth_scale=1000.0):
        """
        从应用了掩码的深度图像中提取3D点
        
        Args:
            depth_image: 深度图像
            mask: 目标掩码
            camera_intrinsics: 相机内参字典 {'fx', 'fy', 'cx', 'cy'}
            depth_scale: 深度缩放因子
            
        Returns:
            points_3d: (N, 3) 形式的3D点坐标数组
            valid_depths: 对应的深度值
        """
        # 应用掩码到深度图像
        masked_depth = self.apply_mask_to_depth(depth_image, mask)
        
        # 将深度值转换为米
        depth_in_meters = masked_depth.astype(np.float32) / depth_scale
        
        # 获取非零深度值的坐标
        valid_indices = np.where(masked_depth > 0)
        y_coords, x_coords = valid_indices
        
        if len(x_coords) == 0:
            return np.empty((0, 3)), np.array([])
        
        # 获取对应的深度值
        valid_depths = depth_in_meters[valid_indices]
        
        # 使用相机内参将像素坐标转换为3D坐标
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # 计算3D坐标
        x_3d = (x_coords - cx) * valid_depths / fx
        y_3d = (y_coords - cy) * valid_depths / fy
        z_3d = valid_depths
        
        # 组合成 (N, 3) 的数组
        points_3d = np.stack((x_3d, y_3d, z_3d), axis=-1)
        
        return points_3d, valid_depths


if __name__ == "__main__":
    # 测试深度处理器
    processor = DepthProcessor()
    
    # 创建测试数据
    depth_img = np.random.randint(0, 4000, (480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 100:200] = 1  # 创建一个矩形掩码
    
    # 测试掩码应用
    masked_depth = processor.apply_mask_to_depth(depth_img, mask)
    print(f"掩码应用完成，原始深度图像形状: {depth_img.shape}, 掩码深度图像形状: {masked_depth.shape}")
    
    # 测试距离过滤
    filtered_depth = processor.filter_depth_by_distance(depth_img, 0.5, 3.0)
    print(f"距离过滤完成")
    
    # 测试提取3D点
    camera_intrinsics = {"fx": 570.3, "fy": 570.3, "cx": 320.0, "cy": 240.0}
    points_3d, depths = processor.extract_points_in_mask(depth_img, mask, camera_intrinsics)
    print(f"提取到 {len(points_3d)} 个3D点")
    
    print("深度处理模块测试完成")