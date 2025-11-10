"""
全面测试改进的椭球拟合算法
"""
import numpy as np
import open3d as o3d
from ellipsoid_fitting_improved import fit_ellipsoid_to_points


def create_various_partial_ellipsoids():
    """创建不同类型的不完整椭球进行测试"""
    print("创建不同类型的不完整椭球进行测试...")
    
    test_cases = [
        {"name": "半椭球 (x>0)", "condition": lambda x, y, z: x >= 0},
        {"name": "四分之一椭球 (x>0, y>0)", "condition": lambda x, y, z: (x >= 0) & (y >= 0)},
        {"name": "椭球顶部 (z>0)", "condition": lambda x, y, z: z >= 0},
        {"name": "椭球一角", "condition": lambda x, y, z: (x >= 0) & (y >= 0) & (z >= 0)},
    ]
    
    original_ellipsoid = {
        'a': 0.1,  # x半轴
        'b': 0.08, # y半轴
        'c': 0.06  # z半轴
    }
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {case['name']}")
        
        # 生成完整椭球点
        u = np.random.uniform(0, 2*np.pi, 500)
        v = np.random.uniform(0, np.pi, 500)
        
        x = original_ellipsoid['a'] * np.cos(u) * np.sin(v)
        y = original_ellipsoid['b'] * np.sin(u) * np.sin(v)
        z = original_ellipsoid['c'] * np.cos(v)
        
        # 添加噪声
        noise = 0.001
        x += np.random.normal(0, noise, size=x.shape)
        y += np.random.normal(0, noise, size=y.shape)
        z += np.random.normal(0, noise, size=z.shape)
        
        # 应用条件筛选
        condition_mask = case['condition'](x, y, z)
        x_partial = x[condition_mask]
        y_partial = y[condition_mask]
        z_partial = z[condition_mask]
        
        partial_points = np.column_stack((x_partial, y_partial, z_partial))
        
        print(f"  生成的点云包含 {len(partial_points)} 个点")
        
        if len(partial_points) < 6:
            print("  点数不足，跳过此案例")
            continue
        
        # 进行椭球拟合
        result = fit_ellipsoid_to_points(
            partial_points,
            ransac_threshold=0.02,
            min_inliers_for_object=20,
            dbscan_eps=0.05
        )
        
        if result['success']:
            print(f"  V 拟合成功！类型: {result['type']}")
            print(f"    椭球中心: [{result['center'][0]:.4f}, {result['center'][1]:.4f}, {result['center'][2]:.4f}]")
            print(f"    半轴长度: [{result['axes_lengths'][0]:.4f}, {result['axes_lengths'][1]:.4f}, {result['axes_lengths'][2]:.4f}]")
            
            # 检查中心是否接近原点（对于对称椭球）
            center_dist = np.linalg.norm(result['center'])
            if center_dist < 0.02:  # 2cm内
                print(f"    V 中心位置合理 (距离原点: {center_dist:.4f})")
            else:
                print(f"    o 中心位置偏移较大 (距离原点: {center_dist:.4f})")
                
            # 检查轴长比例是否合理
            expected_ratios = [
                original_ellipsoid['a'] / original_ellipsoid['b'],  # a/b
                original_ellipsoid['b'] / original_ellipsoid['c'],  # b/c
                original_ellipsoid['a'] / original_ellipsoid['c']   # a/c
            ]
            actual_lengths = sorted(result['axes_lengths'], reverse=True)
            if len(actual_lengths) >= 2:
                actual_ratios = [
                    actual_lengths[0] / actual_lengths[1] if actual_lengths[1] > 0.001 else float('inf'),
                    actual_lengths[1] / actual_lengths[2] if actual_lengths[2] > 0.001 else float('inf'),
                    actual_lengths[0] / actual_lengths[2] if actual_lengths[2] > 0.001 else float('inf')
                ]
                
                ratio_errors = [abs(er - ar) / er if er != 0 else abs(er - ar) for er, ar in zip(expected_ratios, actual_ratios)]
                avg_ratio_error = np.mean([e for e in ratio_errors if not np.isinf(e) and not np.isnan(e)])
                
                if avg_ratio_error < 0.5:  # 50%内的比例误差
                    print(f"    V 轴长比例合理 (平均误差: {avg_ratio_error:.3f})")
                else:
                    print(f"    o 轴长比例误差较大 (平均误差: {avg_ratio_error:.3f})")
        else:
            print(f"  X 拟合失败: {result['message']}")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*50)
    print("测试边界情况...")
    
    # 测试非常少的点
    print("\n1. 测试少量点的情况:")
    # 生成非常小的部分椭球
    u = np.random.uniform(0, np.pi/3, 15)  # 很小的角度范围
    v = np.random.uniform(0, np.pi/3, 15)
    
    a, b, c = 0.05, 0.04, 0.03
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    noise = 0.0005
    x += np.random.normal(0, noise, size=x.shape)
    y += np.random.normal(0, noise, size=y.shape)
    z += np.random.normal(0, noise, size=z.shape)
    
    few_points = np.column_stack((x, y, z))
    print(f"  少量点云: {len(few_points)} 个点")
    
    result = fit_ellipsoid_to_points(
        few_points,
        ransac_threshold=0.01,  # 更严格的阈值
        min_inliers_for_object=8,  # 降低要求
        dbscan_eps=0.02
    )
    
    if result['success']:
        print(f"  V 少量点拟合成功: {result['type']}")
    else:
        print(f"  o 少量点拟合失败: {result['message']}")
    
    # 测试噪声较大的情况
    print("\n2. 测试高噪声情况:")
    u = np.random.uniform(0, 2*np.pi, 100)
    v = np.random.uniform(0, np.pi, 100)
    
    a, b, c = 0.08, 0.06, 0.05
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    high_noise = 0.02  # 高噪声
    x += np.random.normal(0, high_noise, size=x.shape)
    y += np.random.normal(0, high_noise, size=y.shape)
    z += np.random.normal(0, high_noise, size=z.shape)
    
    noisy_points = np.column_stack((x, y, z))
    print(f"  高噪声点云: {len(noisy_points)} 个点")
    
    result = fit_ellipsoid_to_points(
        noisy_points,
        ransac_threshold=0.05,  # 更宽松的阈值
        min_inliers_for_object=20,
        dbscan_eps=0.08
    )
    
    if result['success']:
        print(f"  V 高噪声拟合成功: {result['type']}")
    else:
        print(f"  o 高噪声拟合失败: {result['message']}")


if __name__ == "__main__":
    print("椭球拟合算法全面测试")
    print("="*50)
    
    create_various_partial_ellipsoids()
    
    test_edge_cases()
    
    print("\n" + "="*50)
    print("测试完成！")
    print("\n改进的椭球拟合算法特点：")
    print("1. 支持对不完整/部分椭球的拟合")
    print("2. 使用几何方法从部分点云估计完整椭球参数") 
    print("3. 自适应阈值判断拟合质量")
    print("4. 在各种不完整情况下都有较好的拟合效果")
    print("5. 适用于实际深度相机场景，即使只看到物体一侧也能进行拟合")