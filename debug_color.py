"""
调试颜色转换问题
"""
import numpy as np
import cv2

def debug_color_conversion():
    """调试颜色转换过程"""
    print("调试颜色转换问题...")
    
    # 创建一个简单的测试图像，使用特定颜色
    height, width = 10, 10
    # 假设我们有一个蓝色像素区域 (B=255, G=0, R=0)
    color_img = np.zeros((height, width, 3), dtype=np.uint8)
    color_img[:, :, 0] = 255  # B=255
    color_img[:, :, 1] = 0    # G=0
    color_img[:, :, 2] = 0    # R=0
    
    print(f"原始图像 (BGR) 一个像素: {color_img[0, 0]}")
    print(f"期望RGB: [0, 0, 255] (R=0, G=0, B=255)")
    
    # 模拟处理过程
    mask = np.ones((height, width), dtype=bool)  # 全部像素
    colors_bgr = color_img[mask]  # 提取所有颜色
    print(f"提取的颜色 (BGR): {colors_bgr[0]}")  # 取第一个
    
    # 测试BGR转RGB
    colors_bgr_float = colors_bgr.astype(np.float64)
    print(f"BGR浮点: {colors_bgr_float[0]}")
    
    # 方法1: 使用索引 [2,1,0] 翻转
    colors_rgb_method1 = colors_bgr_float[:, [2, 1, 0]]
    print(f"方法1 [2,1,0] 翻转后: {colors_rgb_method1[0]}")
    
    # 方法2: 使用切片 [::-1] 翻转
    colors_rgb_method2 = colors_bgr_float[:, ::-1]
    print(f"方法2 [::-1] 翻转后: {colors_rgb_method2[0]}")
    
    # 归一化
    colors_rgb_norm = colors_rgb_method1 / 255.0
    print(f"归一化后: {colors_rgb_norm[0]}")
    
    print(f"归一化后的RGB期望值: [0, 0, 1] (R=0, G=0, B=255 -> 1)")
    print(f"实际归一化结果: {colors_rgb_norm[0]}")
    
    # 检查是否匹配
    expected = np.array([0, 0, 255]) / 255.0
    actual = colors_rgb_norm[0]
    print(f"差异: {np.abs(actual - expected)}")
    
    if np.allclose(actual, expected, atol=0.01):
        print("V 颜色转换正确!")
    else:
        print("X 颜色转换错误!")
    
    print("\n" + "="*50)
    
    # 测试原来的测试案例
    print("测试原始测试案例:")
    print("输入BGR [192, 64, 128] 代表 B=192, G=64, R=128")
    print("期望RGB [128, 64, 192] 代表 R=128, G=64, B=192")
    
    test_bgr = np.array([[[192, 64, 128]]], dtype=np.uint8)  # [B, G, R]
    print(f"测试BGR: {test_bgr[0, 0]}")
    
    # 应用转换
    flat_bgr = test_bgr.reshape(-1, 3)
    float_bgr = flat_bgr.astype(np.float64)
    rgb_result = float_bgr[:, [2, 1, 0]]  # [B,G,R] -> [R,G,B]
    rgb_normalized = rgb_result / 255.0
    
    print(f"转换后RGB: {rgb_result[0]}")
    print(f"归一化RGB: {rgb_normalized[0]}")
    print(f"期望归一化RGB: {[128/255.0, 64/255.0, 192/255.0]}")
    
    expected_result = np.array([128, 64, 192]) / 255.0
    if np.allclose(rgb_normalized[0], expected_result, atol=1e-6):
        print("V 颜色转换现在正确!")
    else:
        print("X 颜色转换仍有问题")


if __name__ == "__main__":
    debug_color_conversion()