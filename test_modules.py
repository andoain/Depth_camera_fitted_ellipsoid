"""
测试所有模块是否正确安装和导入
"""
import sys
import os

def test_imports():
    """测试所有模块是否能正确导入"""
    print("开始测试模块导入...")
    
    modules_to_test = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("open3d", "Open3D"),
        ("ultralytics", "Ultralytics (YOLOv8)"),
        ("plotly", "Plotly"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
    ]
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"[OK] {description} ({module_name}) 导入成功")
        except ImportError as e:
            print(f"[FAIL] {description} ({module_name}) 导入失败: {e}")
    
    # 测试自定义模块
    custom_modules = [
        ("pointcloud_utils", "点云工具模块"),
        ("yolo_utils", "YOLO工具模块"),
        ("depth_processing", "深度处理模块"),
        ("ellipsoid_fitting", "椭球拟合模块"),
    ]
    
    for module_name, description in custom_modules:
        try:
            __import__(module_name)
            print(f"[OK] {description} ({module_name}) 导入成功")
        except ImportError as e:
            print(f"[FAIL] {description} ({module_name}) 导入失败: {e}")

def test_functionality():
    """测试各个模块的基本功能"""
    print("\n开始测试模块功能...")
    
    # 测试NumPy
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✅ NumPy 基本功能正常: {arr}")
    except Exception as e:
        print(f"❌ NumPy 功能测试失败: {e}")
    
    # 测试OpenCV
    try:
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        print(f"✅ OpenCV 基本功能正常: 图像形状 {img.shape}")
    except Exception as e:
        print(f"❌ OpenCV 功能测试失败: {e}")
    
    # 测试Open3D
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        print(f"✅ Open3D 基本功能正常: 点云对象创建成功")
    except Exception as e:
        print(f"❌ Open3D 功能测试失败: {e}")
    
    # 测试YOLOv8
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print(f"✅ YOLOv8 基本功能正常: 模型加载成功")
    except Exception as e:
        print(f"❌ YOLOv8 功能测试失败: {e}")
    
    # 测试自定义模块
    try:
        from pointcloud_utils import DepthToPointCloudConverter
        converter = DepthToPointCloudConverter(570.3, 570.3, 320.0, 240.0)
        print(f"[OK] 点云工具模块功能正常: 转换器创建成功")
    except Exception as e:
        print(f"[FAIL] 点云工具模块功能测试失败: {e}")
    
    try:
        from yolo_utils import YOLODetector
        detector = YOLODetector()
        print(f"[OK] YOLO工具模块功能正常: 检测器创建成功")
    except Exception as e:
        print(f"[FAIL] YOLO工具模块功能测试失败: {e}")
    
    try:
        from depth_processing import DepthProcessor
        processor = DepthProcessor()
        print(f"[OK] 深度处理模块功能正常: 处理器创建成功")
    except Exception as e:
        print(f"[FAIL] 深度处理模块功能测试失败: {e}")
    
    try:
        from ellipsoid_fitting import fit_ellipsoid_to_points
        print(f"[OK] 椭球拟合模块功能正常: 函数导入成功")
    except Exception as e:
        print(f"[FAIL] 椭球拟合模块功能测试失败: {e}")

if __name__ == "__main__":
    test_imports()
    test_functionality()
    print("\n模块测试完成！")