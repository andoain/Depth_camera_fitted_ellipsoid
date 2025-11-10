"""
测试Open3D可视化系统
"""
import numpy as np
import open3d as o3d
from improved_visualization import ImprovedVisualizer


def create_test_point_cloud():
    """创建测试点云"""
    # 创建几个椭球形状的点云
    points_list = []
    colors_list = []
    
    # 第一个椭球
    t = np.linspace(0, 2*np.pi, 100)
    s = np.linspace(0, np.pi, 100)
    t, s = np.meshgrid(t, s)
    
    a, b, c = 0.1, 0.05, 0.08
    x = a * np.cos(t) * np.sin(s)
    y = b * np.sin(t) * np.sin(s) 
    z = c * np.cos(s)
    
    # 移动到不同位置
    x += 0.2
    y += 0.1
    z += 0.15
    
    # 添加噪声
    noise = 0.002
    x += np.random.normal(0, noise, x.shape)
    y += np.random.normal(0, noise, y.shape)
    z += np.random.normal(0, noise, z.shape)
    
    points1 = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    colors1 = np.tile([1, 0.7, 0], (len(points1), 1))  # 橙色
    
    # 第二个椭球
    x2 = 0.08 * np.cos(t) * np.sin(s) + 0.2
    y2 = 0.12 * np.sin(t) * np.sin(s) - 0.15
    z2 = 0.06 * np.cos(s) + 0.2
    
    x2 += np.random.normal(0, noise, x2.shape)
    y2 += np.random.normal(0, noise, y2.shape)
    z2 += np.random.normal(0, noise, z2.shape)
    
    points2 = np.column_stack([x2.ravel(), y2.ravel(), z2.ravel()])
    colors2 = np.tile([0, 1, 0], (len(points2), 1))  # 绿色
    
    # 合并所有点
    all_points = np.vstack([points1[:500], points2[:500]])  # 取部分点以加快处理
    all_colors = np.vstack([colors1[:500], colors2[:500]])
    
    return all_points, all_colors


def create_test_fitting_results():
    """创建测试拟合结果"""
    results = []
    
    # 第一个椭球结果
    result1 = {
        'success': True,
        'center': [0.2, 0.1, 0.15],
        'axes_lengths': [0.1, 0.05, 0.08],
        'type': 'Ellipsoid',
        'num_inliers': 500
    }
    results.append(result1)
    
    # 第二个椭球结果
    result2 = {
        'success': True,
        'center': [0.2, -0.15, 0.2],
        'axes_lengths': [0.08, 0.12, 0.06],
        'type': 'Ellipsoid',
        'num_inliers': 480
    }
    results.append(result2)
    
    return results


def main():
    print("测试Open3D可视化系统...")
    
    # 创建测试数据
    points, colors = create_test_point_cloud()
    fitting_results = create_test_fitting_results()
    
    print(f"创建了 {len(points)} 个点的点云")
    print(f"有 {len(fitting_results)} 个椭球拟合结果")
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 使用改进的可视化器
    visualizer = ImprovedVisualizer()
    
    print("开始可视化...")
    print("在可视化窗口中：")
    print("  - 使用鼠标旋转视角")
    print("  - 滚轮缩放")
    print("  - 椭球用不同颜色显示")
    print("  - 红点表示椭球中心")
    
    try:
        visualizer.visualize_all_results(pcd, fitting_results)
        print("可视化完成！")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()