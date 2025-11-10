"""
测试整个场景的椭球拟合功能
"""
import numpy as np
import open3d as o3d
from ellipsoid_fitting_new import fit_ellipsoid_to_points


def test_full_scene_ellipsoid_fitting():
    """测试对整个场景点云进行椭球拟合"""
    print("测试整个场景椭球拟合功能...")
    
    # 模拟一个较大数据集（代表整个场景的点云）
    np.random.seed(42)
    
    # 创建一个主要的椭球形状
    n_points = 1000
    u = np.random.uniform(0, 2*np.pi, n_points)
    v = np.random.uniform(0, np.pi, n_points)
    
    a, b, c = 0.2, 0.15, 0.1  # 椭球半轴
    x_main = a * np.cos(u) * np.sin(v)
    y_main = b * np.sin(u) * np.sin(v)
    z_main = c * np.cos(v)
    
    # 添加噪声
    noise = 0.005
    x_main += np.random.normal(0, noise, size=x_main.shape)
    y_main += np.random.normal(0, noise, size=y_main.shape)
    z_main += np.random.normal(0, noise, size=z_main.shape)
    
    # 添加一些偏移
    x_main += 0.1
    y_main -= 0.05
    z_main += 0.5
    
    # 添加一些背景点（模拟场景中的其他物体）
    n_background = 300
    x_bg = np.random.uniform(-0.3, 0.4, n_background)
    y_bg = np.random.uniform(-0.3, 0.3, n_background)
    z_bg = np.random.uniform(0.3, 0.8, n_background)
    
    # 合并所有点
    x_all = np.concatenate([x_main, x_bg])
    y_all = np.concatenate([y_main, y_bg])
    z_all = np.concatenate([z_main, z_bg])
    
    all_points = np.column_stack((x_all, y_all, z_all))
    
    print(f"整个场景点云包含 {len(all_points)} 个点")
    print(f"点云范围: X[{all_points[:,0].min():.3f}, {all_points[:,0].max():.3f}], "
          f"Y[{all_points[:,1].min():.3f}, {all_points[:,1].max():.3f}], "
          f"Z[{all_points[:,2].min():.3f}, {all_points[:,2].max():.3f}]")
    
    # 进行体素下采样
    print("进行体素下采样...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    voxel_size = 0.01  # 1cm体素大小
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    
    print(f"下采样后点云包含 {len(downsampled_points)} 个点")
    
    # 进行椭球拟合
    print("进行椭球拟合...")
    result = fit_ellipsoid_to_points(
        downsampled_points,
        ransac_threshold=0.02,
        min_inliers_for_object=50,
        dbscan_eps=0.05
    )
    
    if result['success']:
        print("整个场景椭球拟合成功！")
        print(f"椭球类型: {result['type']}")
        print(f"内点数量: {result['num_inliers']}")
        if result['center'] is not None:
            print(f"椭球中心: {result['center']}")
        if result['axes_lengths'] is not None:
            print(f"半轴长度: {result['axes_lengths']}")
        
        # 检查拟合结果是否合理
        expected_center = [0.1, -0.05, 0.5]  # 期望的中心位置
        center_error = np.linalg.norm(np.array(result['center']) - expected_center)
        print(f"中心位置误差: {center_error:.4f}")
        
        if center_error < 0.05:  # 5cm以内为合理误差
            print("V 拟合中心位置合理")
        else:
            print("X 拟合中心位置偏差较大")
            
    else:
        print(f"椭球拟合失败: {result['message']}")


def test_voxel_downsampling():
    """测试体素下采样的效果"""
    print("\n测试体素下采样效果...")
    
    # 创建密集点云
    n_points = 5000
    x = np.random.uniform(-0.1, 0.1, n_points)
    y = np.random.uniform(-0.1, 0.1, n_points)
    z = np.random.uniform(0.5, 0.7, n_points)
    
    # 添加一些重复点和噪声以模拟实际情况
    x = np.concatenate([x, x[:1000] + np.random.normal(0, 0.001, 1000)])  # 重复点
    y = np.concatenate([y, y[:1000] + np.random.normal(0, 0.001, 1000)])
    z = np.concatenate([z, z[:1000] + np.random.normal(0, 0.001, 1000)])
    
    original_points = np.column_stack((x, y, z))
    
    print(f"原始点云包含 {len(original_points)} 个点")
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_points)
    
    # 不同的体素大小测试
    for voxel_size in [0.005, 0.01, 0.02]:
        downsampled = pcd.voxel_down_sample(voxel_size)
        print(f"体素大小 {voxel_size}m: {len(downsampled.points)} 个点")


if __name__ == "__main__":
    print("整个场景椭球拟合功能测试")
    print("="*50)
    
    test_full_scene_ellipsoid_fitting()
    
    test_voxel_downsampling()
    
    print("\n测试完成！")
    print("关键功能：")
    print("1. 检测无目标时自动处理整个场景点云")
    print("2. 对密集点云进行体素下采样")
    print("3. 对下采样后的点云进行椭球拟合")
    print("4. 保存结果并提供可视化")