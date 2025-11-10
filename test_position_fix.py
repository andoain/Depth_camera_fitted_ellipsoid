"""
测试修复后的椭球拟合和颜色处理
"""
import numpy as np
import open3d as o3d
from ellipsoid_fitting_improved import fit_ellipsoid_to_points


def test_ellipsoid_position():
    """测试椭球位置是否正确"""
    print("测试椭球位置修复...")
    
    # 创建一个远离原点的椭球
    offset = [0.5, 0.3, 0.7]  # 偏移量
    a, b, c = 0.1, 0.08, 0.06
    
    # 生成半椭球点（x>offset[0]部分）
    u = np.random.uniform(0, 2*np.pi, 200)
    v = np.random.uniform(0, np.pi, 200)
    
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    # 添加偏移
    x += offset[0]
    y += offset[1]
    z += offset[2]
    
    # 只取一半
    mask = x >= offset[0]
    x_partial = x[mask]
    y_partial = y[mask]
    z_partial = z[mask]
    
    # 添加噪声
    noise = 0.001
    x_partial += np.random.normal(0, noise, size=x_partial.shape)
    y_partial += np.random.normal(0, noise, size=y_partial.shape)
    z_partial += np.random.normal(0, noise, size=z_partial.shape)
    
    points = np.column_stack((x_partial, y_partial, z_partial))
    print(f"生成偏移椭球点云: {len(points)} 个点")
    print(f"预期中心: {offset}")
    
    # 进行椭球拟合
    result = fit_ellipsoid_to_points(
        points,
        ransac_threshold=0.02,
        min_inliers_for_object=20,
        dbscan_eps=0.05
    )
    
    if result['success']:
        print(f"拟合成功！类型: {result['type']}")
        print(f"实际中心: {result['center']}")
        
        # 计算中心偏移
        center_error = np.linalg.norm(np.array(result['center']) - np.array(offset))
        print(f"中心位置偏移: {center_error:.4f}")
        
        if center_error < 0.05:  # 5cm内合理
            print("V 椭球中心位置正确")
        else:
            print("X 椭球中心位置偏移较大")
            
        print(f"轴长: {result['axes_lengths']}")
    else:
        print(f"拟合失败: {result['message']}")


def test_various_positions():
    """测试不同位置的椭球"""
    print("\n测试不同位置的椭球...")
    
    test_positions = [
        [0.0, 0.0, 0.0],    # 原点
        [0.2, 0.0, 0.0],    # x轴方向
        [0.0, 0.3, 0.0],    # y轴方向
        [0.0, 0.0, 0.4],    # z轴方向
        [0.1, 0.2, 0.3],    # 任意位置
        [-0.1, -0.2, -0.3], # 负方向
    ]
    
    for i, pos in enumerate(test_positions):
        print(f"\n测试位置 {i+1}: {pos}")
        
        # 生成椭球点云
        u = np.random.uniform(0, 2*np.pi, 100)
        v = np.random.uniform(0, np.pi, 100)
        
        a, b, c = 0.08, 0.06, 0.05
        x = a * np.cos(u) * np.sin(v) + pos[0]
        y = b * np.sin(u) * np.sin(v) + pos[1]
        z = c * np.cos(v) + pos[2]
        
        # 添加噪声
        noise = 0.001
        x += np.random.normal(0, noise, size=x.shape)
        y += np.random.normal(0, noise, size=y.shape)
        z += np.random.normal(0, noise, size=z.shape)
        
        points = np.column_stack((x, y, z))
        
        result = fit_ellipsoid_to_points(
            points,
            ransac_threshold=0.02,
            min_inliers_for_object=15,
            dbscan_eps=0.03
        )
        
        if result['success']:
            center_error = np.linalg.norm(np.array(result['center']) - np.array(pos))
            print(f"  V 拟合成功，中心误差: {center_error:.4f}")
        else:
            print(f"  X 拟合失败: {result['message']}")


def test_partial_ellipsoids():
    """测试各种部分椭球"""
    print("\n测试各种部分椭球...")
    
    base_points = 150
    shapes = [
        ("半椭球 (x>0)", lambda x, y, z: x > np.mean(x)),
        ("四分之一椭球 (x>0, z>0)", lambda x, y, z: (x > np.mean(x)) & (z > np.mean(z))),
        ("椭球顶部 (z>均值)", lambda x, y, z: z > np.mean(z)),
    ]
    
    a, b, c = 0.1, 0.08, 0.06
    offset = [0.2, -0.1, 0.5]  # 不在原点
    
    for name, condition in shapes:
        print(f"\n测试 {name}:")
        
        # 生成完整椭球
        u = np.random.uniform(0, 2*np.pi, base_points*2)
        v = np.random.uniform(0, np.pi, base_points*2)
        
        x = a * np.cos(u) * np.sin(v) + offset[0]
        y = b * np.sin(u) * np.sin(v) + offset[1]
        z = c * np.cos(v) + offset[2]
        
        # 添加噪声
        noise = 0.001
        x += np.random.normal(0, noise, size=x.shape)
        y += np.random.normal(0, noise, size=y.shape)
        z += np.random.normal(0, noise, size=z.shape)
        
        # 应用条件
        mask = condition(x, y, z)
        x_part = x[mask]
        y_part = y[mask]
        z_part = z[mask]
        
        if len(x_part) < 10:
            print(f"  点数不足，跳过")
            continue
            
        points = np.column_stack((x_part, y_part, z_part))
        print(f"  生成 {len(points)} 个点")
        
        result = fit_ellipsoid_to_points(
            points[:200],  # 限制点数
            ransac_threshold=0.02,
            min_inliers_for_object=15,
            dbscan_eps=0.04
        )
        
        if result['success']:
            center_error = np.linalg.norm(np.array(result['center']) - np.array(offset))
            print(f"  V 拟合成功，中心误差: {center_error:.4f}")
            print(f"    预期中心: {offset}")
            print(f"    实际中心: {result['center']}")
        else:
            print(f"  X 拟合失败: {result['message']}")


if __name__ == "__main__":
    print("测试修复后的椭球拟合和位置问题")
    print("="*50)
    
    test_ellipsoid_position()
    
    test_various_positions()
    
    test_partial_ellipsoids()
    
    print("\n" + "="*50)
    print("测试完成！")
    print("\n修复内容：")
    print("1. 椭球中心位置正确（不再强制置于原点）")
    print("2. 几何椭球拟合使用原始点云的质心")
    print("3. 部分椭球的位置信息保持正确")
    print("4. 适用于实际应用中的任意位置椭球")