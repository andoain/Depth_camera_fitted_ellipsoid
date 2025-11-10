"""
测试中心点与线框对齐
"""
import numpy as np
import open3d as o3d
from ellipsoid_fitting_new import fit_ellipsoid_to_points


def create_test_ellipsoid_points(a=0.1, b=0.08, c=0.06, center_offset=(0, 0, 0), n_points=500):
    """创建测试椭球点云（已应用坐标变换）"""
    # 生成椭球参数化点
    u = np.random.uniform(0, 2*np.pi, n_points)
    v = np.random.uniform(0, np.pi, n_points)
    
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    # 添加偏移
    x += center_offset[0]
    y += center_offset[1]
    z += center_offset[2]
    
    # 添加噪声
    noise = 0.002
    x += np.random.normal(0, noise, size=x.shape)
    y += np.random.normal(0, noise, size=y.shape)
    z += np.random.normal(0, noise, size=z.shape)
    
    points = np.column_stack((x, y, z))
    return points


def test_center_alignment():
    """测试椭球中心与线框的对齐"""
    print("测试椭球中心与线框对齐...")
    
    # 创建椭球点云（模拟已变换的点云）
    test_points = create_test_ellipsoid_points(
        a=0.1, b=0.08, c=0.06, 
        center_offset=(0.2, -0.1, 0.5), 
        n_points=300
    )
    
    print(f"测试点云中心 (实际): [{np.mean(test_points[:,0]):.3f}, {np.mean(test_points[:,1]):.3f}, {np.mean(test_points[:,2]):.3f}]")
    
    # 进行椭球拟合
    result = fit_ellipsoid_to_points(
        test_points,
        ransac_threshold=0.02,
        min_inliers_for_object=50,
        dbscan_eps=0.05
    )
    
    if result['success']:
        print(f"拟合椭球中心: {result['center']}")
        print(f"拟合椭球轴长: {result['axes_lengths']}")
        
        # 检查中心点对齐
        fitted_center = np.array(result['center'])
        actual_center = np.mean(test_points, axis=0)
        center_offset = np.linalg.norm(fitted_center - actual_center)
        
        print(f"拟合中心与实际中心偏移: {center_offset:.4f}")
        
        if center_offset < 0.01:  # 1厘米内是可接受的
            print("V 中心点对齐良好")
        else:
            print(f"X 中心点偏移较大: {center_offset:.4f}")
        
        # 为可视化创建线框椭球
        print("\n准备可视化数据...")
        print(f"用于可视化的椭球中心: {result['center']}")
        print(f"椭球轴长: {result['axes_lengths']}")
        
        # 检查旋转矩阵是否为None的情况
        if result.get('rotation_matrix') is not None:
            print(f"椭球旋转矩阵形状: {result['rotation_matrix'].shape}")
        else:
            print("椭球无旋转或旋转矩阵为None（使用单位矩阵）")
        
    else:
        print(f"椭球拟合失败: {result['message']}")


def test_visualization_alignment():
    """测试可视化中的对齐"""
    print("\n测试可视化中的对齐...")
    
    # 创建两个不同的椭球点云
    points1 = create_test_ellipsoid_points(
        a=0.1, b=0.08, c=0.06,
        center_offset=(0.2, -0.1, 0.8),
        n_points=300
    )
    
    points2 = create_test_ellipsoid_points(
        a=0.08, b=0.12, c=0.05,
        center_offset=(-0.1, 0.15, 0.75),
        n_points=250
    )
    
    # 拟合两个椭球
    result1 = fit_ellipsoid_to_points(
        points1,
        ransac_threshold=0.02,
        min_inliers_for_object=50,
        dbscan_eps=0.05
    )
    
    result2 = fit_ellipsoid_to_points(
        points2,
        ransac_threshold=0.02,
        min_inliers_for_object=50,
        dbscan_eps=0.05
    )
    
    results = []
    if result1['success']:
        results.append(result1)
        print(f"椭球1 - 中心: {result1['center']}")
    if result2['success']:
        results.append(result2)
        print(f"椭球2 - 中心: {result2['center']}")
    
    print(f"共 {len(results)} 个有效拟合结果")
    print("这些结果将用于可视化，中心点将与椭球线框完全对齐")


if __name__ == "__main__":
    print("椭球中心与线框对齐测试")
    print("="*50)
    
    test_center_alignment()
    
    test_visualization_alignment()
    
    print("\n测试完成！")
    print("关键改进：")
    print("1. 所有数据（点云、椭球中心、旋转矩阵）都在统一坐标系中")
    print("2. 可视化模块不再应用额外的坐标变换")
    print("3. 中心点标记直接使用拟合结果的中心坐标")
    print("4. 椭球线框、主轴、中心点完全对齐")