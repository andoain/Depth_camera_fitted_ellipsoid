"""
完整功能测试脚本
"""
import numpy as np
import cv2
from pointcloud_utils import DepthToPointCloudConverter
from yolo_utils import YOLODetector
from depth_processing import DepthProcessor
from ellipsoid_fitting import fit_ellipsoid_to_points


def test_complete_pipeline():
    """测试完整的处理流程"""
    print("开始测试完整处理流程...")
    
    # 1. 初始化各模块
    print("1. 初始化各模块...")
    yolo_detector = YOLODetector(conf_threshold=0.5, iou_threshold=0.5)
    depth_processor = DepthProcessor()
    converter = DepthToPointCloudConverter(
        fx=570.3,
        fy=570.3,
        cx=320.0,
        cy=240.0,
        depth_scale=1000.0
    )
    print("   模块初始化完成")
    
    # 2. 创建模拟数据
    print("2. 创建模拟数据...")
    # 模拟深度图像
    depth_img = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)
    # 模拟彩色图像
    color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # 模拟一个目标掩码
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:300, 200:300] = 1  # 创建一个矩形目标区域
    print(f"   深度图形状: {depth_img.shape}, 彩色图形状: {color_img.shape}")
    
    # 3. 测试深度处理
    print("3. 测试深度处理...")
    filtered_depth = depth_processor.apply_object_mask_and_distance_filter(
        depth_img, 
        mask, 
        min_dist_meters=0.5, 
        max_dist_meters=3.0, 
        depth_scale=1000.0
    )
    print("   深度处理完成")
    
    # 4. 测试彩色点云生成
    print("4. 测试彩色点云生成...")
    pointcloud = converter.convert_with_color(
        filtered_depth,
        color_img,
        min_dist=0.5,
        max_dist=3.0
    )
    print(f"   生成点云，点数: {len(pointcloud.points)}")
    
    # 5. 测试椭球拟合
    print("5. 测试椭球拟合...")
    # 提取3D点
    points_3d, _ = depth_processor.extract_points_in_mask(
        depth_img, 
        mask, 
        {"fx": 570.3, "fy": 570.3, "cx": 320.0, "cy": 240.0}, 
        depth_scale=1000.0
    )
    print(f"   提取到3D点数: {len(points_3d)}")
    
    if len(points_3d) > 9:  # 至少需要9个点进行椭球拟合
        fitting_result = fit_ellipsoid_to_points(
            points_3d,
            ransac_threshold=0.1,
            min_inliers_for_object=10,
            dbscan_eps=0.2
        )
        if fitting_result['success']:
            print(f"   椭球拟合成功！类型: {fitting_result['type']}")
            print(f"   内点数: {fitting_result['num_inliers']}")
        else:
            print(f"   椭球拟合失败: {fitting_result['message']}")
    else:
        print("   点数不足，无法进行椭球拟合")
    
    print("完整处理流程测试完成！")


if __name__ == "__main__":
    test_complete_pipeline()