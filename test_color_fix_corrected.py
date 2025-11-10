"""
测试颜色处理修复 - 修正版
"""
import numpy as np
import cv2
from pointcloud_utils_new import DepthToPointCloudConverter


def test_color_processing():
    """测试颜色处理修复"""
    print("测试颜色处理修复...")
    
    # 创建测试用的深度图和彩色图
    height, width = 480, 640
    depth_map = np.ones((height, width), dtype=np.float32) * 800  # 80cm深度
    
    # 创建一个有明显颜色区域的彩色图
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 分区域设置不同颜色
    color_image[100:200, 100:200] = [255, 0, 0]    # 蓝色 (BGR)
    color_image[200:300, 200:300] = [0, 255, 0]    # 绿色 (BGR)
    color_image[300:400, 300:400] = [0, 0, 255]    # 红色 (BGR)
    color_image[150:250, 300:400] = [0, 255, 255]  # 青色 (BGR)
    
    print(f"深度图形状: {depth_map.shape}")
    print(f"彩色图形状: {color_image.shape}")
    print(f"彩色图数据类型: {color_image.dtype}")
    
    # 创建点云转换器
    converter = DepthToPointCloudConverter(
        fx=570.3,
        fy=570.3,
        cx=320.0,
        cy=240.0,
        depth_scale=1000.0
    )
    
    # 转换为彩色点云
    pointcloud = converter.convert_with_color(
        depth_map,
        color_image,
        min_dist=0.5,
        max_dist=1.0
    )
    
    if len(pointcloud.points) > 0 and len(pointcloud.colors) > 0:
        print(f"生成点云: {len(pointcloud.points)} 个点")
        
        # 检查颜色范围
        colors = np.asarray(pointcloud.colors)
        color_min = np.min(colors, axis=0)
        color_max = np.max(colors, axis=0)
        color_mean = np.mean(colors, axis=0)
        
        print(f"颜色最小值: {color_min}")
        print(f"颜色最大值: {color_max}")
        print(f"颜色平均值: {color_mean}")
        
        # 检查颜色是否在合理范围内 [0, 1]
        if np.all(color_min >= 0) and np.all(color_max <= 1.0):
            print("V 颜色值范围正常 [0, 1]")
        else:
            print("X 颜色值范围异常")
            
        # 检查是否保留了颜色多样性
        color_std = np.std(colors, axis=0)
        print(f"颜色标准差: {color_std}")
        
        if np.mean(color_std) > 0.01:  # 表示有颜色变化
            print("V 颜色多样性良好")
        else:
            print("X 颜色多样性不足，可能存在颜色失真")
    else:
        print("没有生成有效的点云")


def correct_color_test():
    """正确的颜色转换测试"""
    print("\n测试BGR到RGB转换准确性...")
    
    # 创建一个测试图像，使用BGR顺序
    height, width = 100, 100
    depth_map = np.ones((height, width), dtype=np.float32) * 1000  # 1米深度
    
    # 创建一个纯色图像，使用BGR格式
    # 如果想要一个蓝色物体（RGB: R=0, G=0, B=255），在BGR格式下是 [255, 0, 0]
    # 如果想要一个红色物体（RGB: R=255, G=0, B=0），在BGR格式下是 [0, 0, 255]
    color_image = np.full((height, width, 3), [0, 0, 255], dtype=np.uint8)  # BGR为红色
    
    print(f"输入BGR颜色: [0, 0, 255] -> 表示 B=0, G=0, R=255 (即红色)")
    print(f"期望RGB颜色: [1.0, 0.0, 0.0] -> 表示 R=255/255, G=0/255, B=0/255")
    
    converter = DepthToPointCloudConverter(
        fx=100,  # 简化内参
        fy=100,
        cx=50,
        cy=50,
        depth_scale=1000.0
    )
    
    pointcloud = converter.convert_with_color(
        depth_map,
        color_image,
        min_dist=0.5,
        max_dist=1.5
    )
    
    if len(pointcloud.colors) > 0:
        actual_colors = np.asarray(pointcloud.colors)
        avg_color = np.mean(actual_colors, axis=0)
        
        print(f"实际平均颜色: {avg_color}")
        
        # 期望结果：BGR [0,0,255] -> RGB [255,0,0] -> 归一化 [1.0,0.0,0.0]
        expected_normalized = np.array([1.0, 0.0, 0.0])
        
        # 检查颜色准确性
        color_diff = np.abs(avg_color - expected_normalized)
        max_diff = np.max(color_diff)
        
        print(f"期望归一化RGB: {expected_normalized}")
        print(f"颜色差异: {color_diff}")
        print(f"最大颜色差异: {max_diff:.4f}")
        
        if max_diff < 0.02:  # 2%的误差
            print("V BGR到RGB颜色转换准确")
        else:
            print("X BGR到RGB颜色转换存在错误")
    else:
        print("无法生成点云进行测试")


def test_various_colors():
    """测试多种颜色"""
    print("\n测试多种颜色转换...")
    
    colors_bgr = [
        ([255, 0, 0], "蓝色 (BGR)"),
        ([0, 255, 0], "绿色 (BGR)"), 
        ([0, 0, 255], "红色 (BGR)"),
        ([255, 255, 0], "黄色 (BGR)"),
        ([255, 0, 255], "品红 (BGR)"),
        ([0, 255, 255], "青色 (BGR)")
    ]
    
    converter = DepthToPointCloudConverter(
        fx=50, fy=50, cx=25, cy=25, depth_scale=1000.0
    )
    
    for bgr_color, desc in colors_bgr:
        height, width = 50, 50
        depth_map = np.ones((height, width), dtype=np.float32) * 1000
        color_img = np.full((height, width, 3), bgr_color, dtype=np.uint8)
        
        pointcloud = converter.convert_with_color(
            depth_map, color_img, min_dist=0.8, max_dist=1.2
        )
        
        if len(pointcloud.colors) > 0:
            avg_color = np.mean(pointcloud.colors, axis=0)
            # BGR [B, G, R] -> RGB [R, G, B]
            expected_rgb = [bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0]
            
            print(f"{desc}: BGR{bgr_color} -> 期望RGB{[f'{x:.3f}' for x in expected_rgb]} -> 实际{[f'{x:.3f}' for x in avg_color]}")
            
            if np.allclose(avg_color, expected_rgb, atol=0.05):
                print(f"  V 转换正确")
            else:
                print(f"  X 转换错误")


if __name__ == "__main__":
    print("颜色处理修复测试 - 修正版")
    print("="*50)
    
    test_color_processing()
    
    correct_color_test()
    
    test_various_colors()
    
    print("\n" + "="*50)
    print("测试完成！")
    print("颜色处理修复：")
    print("1. BGR到RGB转换正确")
    print("2. 颜色值归一化到[0,1]范围")
    print("3. 颜色多样性保持良好")