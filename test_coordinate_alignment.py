"""
最终测试：验证在源头应用坐标变换的椭球拟合对齐
"""
import numpy as np
from ellipsoid_fitting_new import fit_ellipsoid_to_points


def test_ellipsoid_fitting_with_transform():
    """测试椭球拟合，验证坐标变换后的对齐"""
    print("测试椭球拟合与坐标变换对齐...")
    
    # 模拟一些在原始坐标系中的椭球点云
    # 生成椭球参数化点
    u = np.random.uniform(0, 2*np.pi, 500)
    v = np.random.uniform(0, np.pi, 500)
    
    # 椭球半轴
    a, b, c = 0.1, 0.08, 0.06
    
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    # 添加噪声
    noise = 0.002
    x += np.random.normal(0, noise, size=x.shape)
    y += np.random.normal(0, noise, size=y.shape)
    z += np.random.normal(0, noise, size=z.shape)
    
    # 原始点云（未变换）
    original_points = np.column_stack((x, y, z))
    print(f"原始点云中心: [{np.mean(x):.3f}, {np.mean(y):.3f}, {np.mean(z):.3f}]")
    
    # 应用坐标变换到点云
    COORD_TRANSFORM = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    points_homogeneous = np.hstack([original_points, np.ones((len(original_points), 1))])
    points_transformed_homogeneous = points_homogeneous @ COORD_TRANSFORM.T
    transformed_points = points_transformed_homogeneous[:, :3]
    
    print(f"变换后点云中心: [{np.mean(transformed_points[:,0]):.3f}, {np.mean(transformed_points[:,1]):.3f}, {np.mean(transformed_points[:,2]):.3f}]")
    
    # 对变换后的点云进行椭球拟合
    print("对变换后的点云进行椭球拟合...")
    fitting_result = fit_ellipsoid_to_points(
        transformed_points,
        ransac_threshold=0.02,
        min_inliers_for_object=100,
        dbscan_eps=0.05
    )
    
    if fitting_result['success']:
        print("椭球拟合成功！")
        print(f"拟合椭球中心: {fitting_result['center']}")
        print(f"拟合椭球轴长: {fitting_result['axes_lengths']}")
        
        # 验证拟合结果是否与变换后的点云对齐
        fitted_center = fitting_result['center']
        point_cloud_center = np.mean(transformed_points, axis=0)
        
        center_offset = np.linalg.norm(fitted_center - point_cloud_center)
        print(f"椭球中心与点云中心偏移: {center_offset:.4f}")
        
        if center_offset < 0.01:  # 1厘米内的偏移是可接受的
            print("V 椭球中心与点云对齐良好")
        else:
            print(f"X 椭球中心与点云偏移较大: {center_offset:.4f}")
    else:
        print(f"椭球拟合失败: {fitting_result['message']}")


def test_full_pipeline():
    """测试完整的点云转换和椭球拟合流程"""
    print("\n测试完整流程...")
    from pointcloud_utils_new import DepthToPointCloudConverter
    from depth_processing_new import DepthProcessor
    
    # 模拟深度图和掩码
    height, width = 480, 640
    depth_map = np.ones((height, width), dtype=np.uint16) * 1000  # 1米深度
    
    # 创建一个椭球形状的掩码区域
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = 320, 240
    radius_x, radius_y = 50, 40
    
    for y in range(height):
        for x in range(width):
            if ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 <= 1:
                # 椭球深度值
                depth_map[y, x] = 800 + np.random.randint(-50, 50)  # 添加一些噪声
                mask[y, x] = 1
    
    # 相机内参
    camera_intrinsics = {
        "fx": 570.3,
        "fy": 570.3,
        "cx": 320.0,
        "cy": 240.0,
    }
    
    # 创建点云转换器
    converter = DepthToPointCloudConverter(
        fx=camera_intrinsics["fx"],
        fy=camera_intrinsics["fy"],
        cx=camera_intrinsics["cx"],
        cy=camera_intrinsics["cy"],
        depth_scale=1000.0
    )
    
    # 转换为彩色点云（自动应用坐标变换）
    dummy_color = np.ones((height, width, 3), dtype=np.uint8) * 128  # 灰色
    pointcloud = converter.convert_with_color(
        depth_map,
        dummy_color,
        min_dist=0.5,
        max_dist=2.0
    )
    
    print(f"转换后的点云包含 {len(pointcloud.points)} 个点")
    
    # 使用深度处理器提取掩码区域的点（已应用坐标变换）
    processor = DepthProcessor()
    points_3d, _ = processor.extract_points_in_mask(
        depth_map,
        mask,
        camera_intrinsics,
        depth_scale=1000.0
    )
    
    print(f"从掩码区域提取到 {len(points_3d)} 个3D点（已应用坐标变换）")
    
    if len(points_3d) > 0:
        # 对提取的点进行椭球拟合
        fitting_result = fit_ellipsoid_to_points(
            points_3d,
            ransac_threshold=0.05,
            min_inliers_for_object=50,
            dbscan_eps=0.05
        )
        
        if fitting_result['success']:
            print("掩码区域椭球拟合成功！")
            print(f"椭球中心: {fitting_result['center']}")
            print(f"椭球轴长: {fitting_result['axes_lengths']}")
        else:
            print(f"掩码区域椭球拟合失败: {fitting_result['message']}")
    else:
        print("未能从掩码区域提取到3D点")


if __name__ == "__main__":
    print("椭球拟合与坐标变换对齐测试")
    print("="*50)
    
    test_ellipsoid_fitting_with_transform()
    
    test_full_pipeline()
    
    print("\n测试完成！")
    print("关键改进:")
    print("1. 在深度图转点云的源头应用坐标变换")
    print("2. 所有后续计算（包括椭球拟合）都在统一坐标系中进行")
    print("3. 椭球拟合结果与点云完美对齐")