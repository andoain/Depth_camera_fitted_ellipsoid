"""
测试高级可视化模块
"""
import numpy as np
import open3d as o3d
import cv2
from advanced_visualization import AdvancedVisualizer


def create_test_depth_and_color():
    """创建测试用的深度图和彩色图"""
    height, width = 480, 640
    
    # 创建测试深度图（模拟两个物体）
    depth = np.ones((height, width), dtype=np.float32) * 1000  # 背景深度1米
    
    # 第一个物体（椭球形状）
    center1_x, center1_y = 200, 150
    radius1_x, radius1_y = 80, 60
    for y in range(height):
        for x in range(width):
            if ((x - center1_x) / radius1_x)**2 + ((y - center1_y) / radius1_y)**2 <= 1:
                depth[y, x] = 800  # 距离相机0.8米
    
    # 第二个物体
    center2_x, center2_y = 400, 300
    radius2_x, radius2_y = 60, 90
    for y in range(height):
        for x in range(width):
            if ((x - center2_x) / radius2_x)**2 + ((y - center2_y) / radius2_y)**2 <= 1:
                depth[y, x] = 750  # 距离相机0.75米
    
    # 创建彩色图（模拟彩色信息）
    color = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # 为不同区域设置不同颜色
    for y in range(height):
        for x in range(width):
            if ((x - center1_x) / radius1_x)**2 + ((y - center1_y) / radius1_y)**2 <= 1:
                color[y, x] = [255, 100, 100]  # 红色区域
            elif ((x - center2_x) / radius2_x)**2 + ((y - center2_y) / radius2_y)**2 <= 1:
                color[y, x] = [100, 255, 100]  # 绿色区域
    
    return depth, color


def create_test_fitting_results():
    """创建测试拟合结果"""
    results = []
    
    # 第一个椭球结果
    result1 = {
        'success': True,
        'center': [0.2, -0.1, 0.8],  # 根据相机内参计算的3D位置
        'axes_lengths': [0.1, 0.08, 0.06],
        'type': 'Ellipsoid',
        'num_inliers': 500,
        'rotation_matrix': np.eye(3)  # 单位矩阵表示无旋转
    }
    results.append(result1)
    
    # 第二个椭球结果
    result2 = {
        'success': True,
        'center': [-0.1, 0.15, 0.75],
        'axes_lengths': [0.08, 0.12, 0.05],
        'type': 'Ellipsoid',
        'num_inliers': 450,
        'rotation_matrix': np.eye(3)
    }
    results.append(result2)
    
    return results


def test_advanced_visualization():
    """测试高级可视化"""
    print("创建测试数据...")
    
    # 创建测试深度图和彩色图
    depth_img, color_img = create_test_depth_and_color()
    fitting_results = create_test_fitting_results()
    
    # 相机内参
    camera_intrinsics = {
        "fx": 570.3,
        "fy": 570.3,
        "cx": 320.0,
        "cy": 240.0,
    }
    
    print(f"深度图形状: {depth_img.shape}")
    print(f"彩色图形状: {color_img.shape}")
    print(f"拟合结果数量: {len(fitting_results)}")
    
    # 创建可视化器
    visualizer = AdvancedVisualizer()
    
    print("\n开始可视化测试...")
    print("在可视化窗口中：")
    print("  - 彩色点云会显示真实颜色")
    print("  - 半透明椭球显示拟合结果")
    print("  - 红点标记椭球中心")
    print("  - 坐标系显示椭球方向")
    
    try:
        # 使用完整场景可视化
        visualizer.visualize_fitting_results_with_colored_pointcloud(
            depth_img,
            color_img,
            camera_intrinsics,
            fitting_results
        )
        print("高级可视化测试完成！")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def test_single_object_visualization():
    """测试单个物体可视化"""
    print("\n" + "="*50)
    print("测试单个物体可视化...")
    
    # 创建测试数据
    depth_img, color_img = create_test_depth_and_color()
    single_result = create_test_fitting_results()[0]  # 只取第一个结果
    
    camera_intrinsics = {
        "fx": 570.3,
        "fy": 570.3,
        "cx": 320.0,
        "cy": 240.0,
    }
    
    # 创建一个简单的掩码
    mask = np.zeros(depth_img.shape, dtype=np.uint8)
    center_x, center_y = 200, 150
    radius_x, radius_y = 80, 60
    for y in range(depth_img.shape[0]):
        for x in range(depth_img.shape[1]):
            if ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 <= 1:
                mask[y, x] = 1
    
    visualizer = AdvancedVisualizer()
    
    print("启动单物体可视化...")
    try:
        visualizer.visualize_single_object_with_ellipsoid(
            depth_img,
            color_img,
            camera_intrinsics,
            single_result,
            mask=mask
        )
        print("单物体可视化测试完成！")
    except Exception as e:
        print(f"单物体可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("高级Open3D可视化测试")
    print("="*50)
    
    test_advanced_visualization()
    
    test_single_object_visualization()
    
    print("\n所有测试完成！")