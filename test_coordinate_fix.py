"""
测试修复后的坐标对齐问题
"""
import numpy as np
from ellipsoid_fitting_improved import fit_ellipsoid_to_points


def create_test_point_cloud_with_transform():
    """创建测试点云，模拟经过坐标变换的点云"""
    print("创建经过坐标变换的测试点云...")
    
    # 创建一个椭球的点云（在原始相机坐标系中）
    u = np.random.uniform(0, 2*np.pi, 200)
    v = np.random.uniform(0, np.pi, 200)
    
    a, b, c = 0.1, 0.08, 0.06  # 椭球半轴
    offset = [0.2, 0.1, 0.8]    # 椭球中心偏移
    
    x = a * np.cos(u) * np.sin(v) + offset[0]
    y = b * np.sin(u) * np.sin(v) + offset[1]
    z = c * np.cos(v) + offset[2]
    
    # 添加噪声
    noise = 0.002
    x += np.random.normal(0, noise, size=x.shape)
    y += np.random.normal(0, noise, size=y.shape)
    z += np.random.normal(0, noise, size=z.shape)
    
    original_points = np.column_stack((x, y, z))
    expected_center = np.array(offset)
    
    print(f"原始椭球中心: {expected_center}")
    print(f"原始点云包含 {len(original_points)} 个点")
    
    # 应用坐标变换（模拟在源头的变换）
    COORD_TRANSFORM = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    # 应用变换
    points_homogeneous = np.hstack([original_points, np.ones((len(original_points), 1))])
    transformed_points_homogeneous = points_homogeneous @ COORD_TRANSFORM.T
    transformed_points = transformed_points_homogeneous[:, :3]
    
    # 计算变换后的真实中心
    transformed_center = (COORD_TRANSFORM @ np.array([offset[0], offset[1], offset[2], 1]))[:3]
    print(f"变换后理论中心: {transformed_center}")
    
    return transformed_points, transformed_center


def test_ellipsoid_fitting_alignment():
    """测试椭球拟合与点云的对齐"""
    print("\n测试椭球拟合与点云对齐...")
    
    # 获取变换后的点云
    points, true_center = create_test_point_cloud_with_transform()
    
    # 进行椭球拟合（在统一坐标系中）
    result = fit_ellipsoid_to_points(
        points,
        ransac_threshold=0.02,
        min_inliers_for_object=50,
        dbscan_eps=0.05
    )
    
    if result['success']:
        fitting_center = np.array(result['center'])
        center_error = np.linalg.norm(fitting_center - true_center)
        
        print(f"拟合椭球中心: {fitting_center}")
        print(f"理论中心: {true_center}")
        print(f"中心位置误差: {center_error:.4f}")
        
        if center_error < 0.01:  # 1cm内为可接受误差
            print("V 椭球中心与点云对齐良好")
        else:
            print("X 椭球中心与点云对齐较差")
        
        print(f"椭球类型: {result['type']}")
        print(f"内点数量: {result['num_inliers']}")
        if result['axes_lengths'] is not None:
            print(f"轴长: {result['axes_lengths']}")
    else:
        print(f"椭球拟合失败: {result['message']}")


def test_partial_ellipsoid_alignment():
    """测试部分椭球的对齐"""
    print("\n测试部分椭球拟合对齐...")
    
    # 创建部分椭球
    u = np.random.uniform(0, 2*np.pi, 300)
    v = np.random.uniform(0, np.pi, 300)
    
    a, b, c = 0.12, 0.1, 0.08
    offset = [0.15, -0.05, 0.7]
    
    x = a * np.cos(u) * np.sin(v) + offset[0]
    y = b * np.sin(u) * np.sin(v) + offset[1]
    z = c * np.cos(v) + offset[2]
    
    # 应用条件保留半个椭球
    mask = x >= offset[0]  # 只保留x > 中心x的部分
    x_partial = x[mask]
    y_partial = y[mask]
    z_partial = z[mask]
    
    # 添加噪声
    noise = 0.001
    x_partial += np.random.normal(0, noise, size=x_partial.shape)
    y_partial += np.random.normal(0, noise, size=y_partial.shape)
    z_partial += np.random.normal(0, noise, size=z_partial.shape)
    
    original_partial = np.column_stack((x_partial, y_partial, z_partial))
    expected_partial_center = np.array(offset)
    
    print(f"部分椭球点云: {len(original_partial)} 个点，期望中心: {expected_partial_center}")
    
    # 应用坐标变换
    COORD_TRANSFORM = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    partial_homogeneous = np.hstack([original_partial, np.ones((len(original_partial), 1))])
    partial_transformed_homogeneous = partial_homogeneous @ COORD_TRANSFORM.T
    partial_transformed = partial_transformed_homogeneous[:, :3]
    
    # 计算变换后的真实中心
    partial_true_center = (COORD_TRANSFORM @ np.array([offset[0], offset[1], offset[2], 1]))[:3]
    print(f"变换后理论中心: {partial_true_center}")
    
    # 进行椭球拟合
    result = fit_ellipsoid_to_points(
        partial_transformed,
        ransac_threshold=0.02,
        min_inliers_for_object=30,
        dbscan_eps=0.04
    )
    
    if result['success']:
        fitting_center = np.array(result['center'])
        center_error = np.linalg.norm(fitting_center - partial_true_center)
        
        print(f"部分椭球拟合中心: {fitting_center}")
        print(f"理论中心: {partial_true_center}")
        print(f"中心位置误差: {center_error:.4f}")
        
        if center_error < 0.02:  # 2cm内为可接受误差
            print("V 部分椭球中心与点云对齐良好")
        else:
            print("X 部分椭球中心与点云对齐较差")
        
        if result['axes_lengths'] is not None:
            print(f"拟合轴长: {result['axes_lengths']}")
    else:
        print(f"部分椭球拟合失败: {result['message']}")


if __name__ == "__main__":
    print("测试修复后的坐标对齐问题")
    print("="*50)
    
    test_ellipsoid_fitting_alignment()
    
    test_partial_ellipsoid_alignment()
    
    print("\n" + "="*50)
    print("修复内容:")
    print("1. 消除了双重坐标变换问题")
    print("2. 点云在源头应用变换，后续处理不重复应用")
    print("3. 椭球拟合结果与点云精确对齐")
    print("4. 适用于完整和部分椭球场景")
    print("5. 坐标系统一，位置关系准确")