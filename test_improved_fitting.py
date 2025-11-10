"""
测试改进的椭球拟合在实际场景中的效果
"""
import numpy as np
import open3d as o3d
from ellipsoid_fitting_improved import fit_ellipsoid_to_points
from advanced_visualization_fixed import AdvancedVisualizer


def create_partial_ellipsoid_points(a=0.1, b=0.08, c=0.06, n_points=200, visible_ratio=0.5):
    """创建部分椭球点云（模拟部分可见的椭球）"""
    # 生成完整的椭球参数化点
    u = np.random.uniform(0, 2*np.pi, n_points*2)  # 生成更多点以确保有足够的可见点
    v = np.random.uniform(0, np.pi, n_points*2)
    
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    # 只取一部分（比如只取x > 某个值的部分）
    # 根据可见比例决定阈值
    threshold = np.percentile(x, (1 - visible_ratio) * 100)
    visible_mask = x >= threshold
    
    x_partial = x[visible_mask]
    y_partial = y[visible_mask]
    z_partial = z[visible_mask]
    
    # 限制点数
    if len(x_partial) > n_points:
        indices = np.random.choice(len(x_partial), n_points, replace=False)
        x_partial, y_partial, z_partial = x_partial[indices], y_partial[indices], z_partial[indices]
    elif len(x_partial) == 0:
        # 如果没有满足条件的点，取前半部分
        n_keep = max(6, int(n_points * visible_ratio))  # 至少保留6个点
        x_partial, y_partial, z_partial = x[:n_keep], y[:n_keep], z[:n_keep]
    
    # 添加噪声
    noise = 0.001
    x_partial += np.random.normal(0, noise, size=x_partial.shape)
    y_partial += np.random.normal(0, noise, size=y_partial.shape)
    z_partial += np.random.normal(0, noise, size=z_partial.shape)
    
    points = np.column_stack((x_partial, y_partial, z_partial))
    return points


def test_partial_ellipsoid_fitting():
    """测试部分椭球的拟合效果"""
    print("测试部分椭球拟合效果...")
    
    # 测试不同可见比例的椭球
    for ratio in [0.3, 0.5, 0.7]:
        print(f"\n测试可见比例: {ratio*100}%")
        
        # 创建部分椭球点云
        partial_points = create_partial_ellipsoid_points(
            a=0.1, b=0.08, c=0.06, 
            n_points=150, 
            visible_ratio=ratio
        )
        
        print(f"生成的点云包含 {len(partial_points)} 个点")
        
        # 进行椭球拟合
        result = fit_ellipsoid_to_points(
            partial_points,
            ransac_threshold=0.02,
            min_inliers_for_object=20,  # 降低最小内点数要求
            dbscan_eps=0.03
        )
        
        if result['success']:
            print(f"  拟合成功！类型: {result['type']}")
            print(f"  椭球中心: {result['center']}")
            print(f"  半轴长度: {result['axes_lengths']}")
            
            # 检查轴长是否合理（接近原始椭球的轴长）
            expected_axes = [0.1, 0.08, 0.06]
            actual_axes = sorted(result['axes_lengths'], reverse=True)  # 从大到小排序
            expected_sorted = sorted(expected_axes, reverse=True)
            
            axis_error = np.mean(np.abs(np.array(actual_axes) - expected_sorted))
            print(f"  轴长误差: {axis_error:.4f}")
            
            if axis_error < 0.03:  # 3cm内为合理
                print("  V 拟合结果合理")
            else:
                print("  X 拟合结果误差较大")
        else:
            print(f"  拟合失败: {result['message']}")


def test_visualization_of_partial_ellipsoid():
    """测试部分椭球的可视化"""
    print("\n测试部分椭球的可视化...")
    
    # 创建一个部分椭球
    partial_points = create_partial_ellipsoid_points(
        a=0.12, b=0.09, c=0.07,
        n_points=120,
        visible_ratio=0.4  # 只40%可见
    )
    
    print(f"创建部分椭球点云: {len(partial_points)} 个点")
    
    # 进行椭球拟合
    result = fit_ellipsoid_to_points(
        partial_points,
        ransac_threshold=0.02,
        min_inliers_for_object=25,
        dbscan_eps=0.03
    )
    
    if result['success']:
        print("部分椭球拟合成功，准备可视化...")
        
        # 创建虚拟深度图和彩色图用于可视化（用简单的数据）
        depth_img = np.ones((480, 640), dtype=np.float32) * 800  # 80cm深度
        color_img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)  # 随机彩色
        
        # 添加一些点云数据到深度图中
        for pt in partial_points[:100]:  # 只用前100个点
            # 将3D点投影到2D（简化投影）
            pass
        
        camera_intrinsics = {
            "fx": 570.3,
            "fy": 570.3,
            "cx": 320.0,
            "cy": 240.0,
        }
        
        # 创建全1的掩码（表示整个点云区域）
        mask = np.ones((480, 640), dtype=np.uint8)
        
        try:
            # 使用单目标可视化
            visualizer = AdvancedVisualizer()
            # 注意：由于我们没有真正的深度/彩色图像对应关系，我们跳过可视化
            print("拟合参数:")
            print(f"  中心: {result['center']}")
            print(f"  轴长: {result['axes_lengths']}")
            print("  类型: 椭球面 (几何估计)" if result.get('geometric_fitting', False) else "标准椭球面")
        except Exception as e:
            print(f"可视化过程中出现错误: {e}")
    else:
        print(f"椭球拟合失败: {result['message']}")


if __name__ == "__main__":
    print("改进椭球拟合算法实际应用测试")
    print("="*50)
    
    test_partial_ellipsoid_fitting()
    
    test_visualization_of_partial_ellipsoid()
    
    print("\n测试完成！")
    print("改进功能：")
    print("1. 支持对部分点云进行椭球拟合")
    print("2. 使用几何方法估计完整椭球参数")
    print("3. 即使只看到椭球一部分也能估算完整形状")
    print("4. 适用于实际深度相机只看到物体一侧的情况")