"""
完整功能的主程序：处理深度相机数据，YOLO分割，彩色点云重建，椭球拟合和可视化
"""
import os
import sys
import numpy as np
import cv2
from pointcloud_utils_new import DepthToPointCloudConverter
from yolo_utils import YOLODetector
from depth_processing_fixed import DepthProcessor
from ellipsoid_fitting_improved import fit_ellipsoid_to_points
from advanced_visualization_fixed import visualize_all_objects_with_colored_pointcloud


# 设置OpenNI库路径 - 确保OpenNI2.dll在系统路径中
openni_sdk_path = os.path.join(os.path.dirname(__file__), "openniSDKWin64", "sdk", "libs")
if os.path.exists(openni_sdk_path):
    # 添加库路径到环境变量（Windows）
    current_path = os.environ.get('PATH', '')
    if openni_sdk_path not in current_path:
        os.environ['PATH'] = current_path + os.pathsep + openni_sdk_path
    # 也添加OpenNI2子目录（包含驱动）
    openni2_path = os.path.join(openni_sdk_path, "OpenNI2")
    if os.path.exists(openni2_path) and openni2_path not in current_path:
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + openni2_path

try:
    from openni import openni2
    OPENNI_AVAILABLE = True
except ImportError:
    print("警告: OpenNI库未安装或不可用，将使用模拟模式")
    OPENNI_AVAILABLE = False


# ------------------------------------------------------------------
# 1. 相机内参
# ------------------------------------------------------------------
CAMERA_INTRINSICS = {
    "fx": 570.3,
    "fy": 570.3,
    "cx": 320.0,
    "cy": 240.0,
}
DEPTH_SCALE = 1000.0

FILTER_MIN_DIST = 0.55  # (米) 过滤掉小于 0.55m 的点
FILTER_MAX_DIST = 8.0  # (米) 过滤掉大于 8.0m 的点 (相机的标称范围)

# YOLO模型参数
YOLO_MODEL_PATH = 'yolov8n.pt'  # 可以根据需要更换为其他模型如 'yolov8n-seg.pt'
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# 椭球拟合参数
ELLIPSOID_RANSAC_THRESHOLD = 0.02
ELLIPSOID_MIN_INLIERS = 50
ELLIPSOID_DBSCAN_EPS = 0.05


class DepthCameraProcessor:
    """
    深度相机处理器：整合所有功能模块
    """
    def __init__(self):
        if not OPENNI_AVAILABLE:
            print("错误: OpenNI库不可用，无法初始化深度相机。")
            raise RuntimeError("OpenNI库不可用")

        # 初始化各模块
        self.yolo_detector = YOLODetector(YOLO_MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)
        self.depth_processor = DepthProcessor()
        self.converter = DepthToPointCloudConverter(
            fx=CAMERA_INTRINSICS["fx"],
            fy=CAMERA_INTRINSICS["fy"],
            cx=CAMERA_INTRINSICS["cx"],
            cy=CAMERA_INTRINSICS["cy"],
            depth_scale=DEPTH_SCALE
        )

        # 初始化相机
        print("尝试初始化OpenNI...")
        openni_sdk_path = os.path.join(os.path.dirname(__file__), "openniSDKWin64", "sdk", "libs")
        if os.path.exists(openni_sdk_path):
            openni2.initialize(openni_sdk_path)
        else:
            print(f"警告: 找不到OpenNI SDK路径: {openni_sdk_path}")
            openni2.initialize()

        print("正在打开设备...")
        self.dev = openni2.Device.open_any()
        print("设备打开成功")
        print(self.dev.get_device_info())
        self.depth_stream = self.dev.create_depth_stream()
        self.dev.set_image_registration_mode(True)
        self.depth_stream.start()

        # 初始化彩色摄像头
        self.cap = cv2.VideoCapture(0)

    def process_frame(self):
        """
        处理单帧数据：获取深度图和彩色图，进行目标检测和点云生成
        """
        # 获取深度帧
        frame = self.depth_stream.read_frame()
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')

        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt = cv2.flip(dpt, 1)
        dpt = dpt.astype(np.uint16)

        # 获取彩色帧
        ret, color_frame = self.cap.read()

        if not ret:
            return None, None, None, None, None

        # 使用YOLO进行目标检测
        masks, boxes, class_ids = self.yolo_detector.get_detection_masks(color_frame)

        # 如果没有检测到任何目标，设置一个标志
        has_detections = len(masks) > 0
       

        # 显示深度图
        dim_gray = cv2.convertScaleAbs(dpt, alpha=0.0425)
        depth_colormap = cv2.applyColorMap(dim_gray, 2)
        cv2.imshow('depth', depth_colormap)

        # 显示彩色图与检测结果
        annotated_img = self.yolo_detector.draw_detections(color_frame, masks, boxes, class_ids)
        cv2.imshow('color', annotated_img)

        return dpt, color_frame, masks, boxes, class_ids

    def extract_object_pointcloud(self, depth_image, color_image, mask):
        """
        从指定掩码区域提取彩色点云
        """
        # 应用掩码和距离过滤到深度图
        filtered_depth = self.depth_processor.apply_object_mask_and_distance_filter(
            depth_image,
            mask,
            FILTER_MIN_DIST,
            FILTER_MAX_DIST,
            DEPTH_SCALE
        )

        # 转换为彩色点云
        pointcloud = self.converter.convert_with_color(
            filtered_depth,
            color_image,
            FILTER_MIN_DIST,
            FILTER_MAX_DIST
        )

        return pointcloud

    def fit_ellipsoid_to_object(self, depth_image, color_image, mask):
        """
        对检测到的目标进行椭球拟合
        """
        # 提取目标区域的3D点
        points_3d, _ = self.depth_processor.extract_points_in_mask(
            depth_image,
            mask,
            CAMERA_INTRINSICS,
            DEPTH_SCALE
        )

        if len(points_3d) == 0:
            print("警告: 目标区域内没有有效深度点")
            return None, None

        # 进行椭球拟合
        fitting_result = fit_ellipsoid_to_points(
            points_3d,
            ransac_threshold=ELLIPSOID_RANSAC_THRESHOLD,
            min_inliers_for_object=ELLIPSOID_MIN_INLIERS,
            dbscan_eps=ELLIPSOID_DBSCAN_EPS
        )

        return fitting_result, points_3d

    def run(self):
        """
        运行主循环
        """
        cv2.namedWindow('depth')
        cv2.namedWindow('color')

        capture_flag = 0
        base_path = 'dataset/'
        count = 1

        print("程序已启动")
        print("按 'q' 键退出")
        print("按 'r' 键处理当前帧")

        while True:
            dpt, color_frame, masks, boxes, class_ids = self.process_frame()

            if dpt is None or color_frame is None:
                continue

            key = cv2.waitKey(30)
            if int(key) == ord('q'):
                break
            if int(key) == ord('r'):
                capture_flag = 1

            if capture_flag == 1 and masks is not None:
                print(f"正在处理第 {count} 帧...")

                # 存储所有目标的拟合结果
                all_fitting_results = []
                all_points_3d = []
                
                # 对每个检测到的目标进行处理
                for i, mask in enumerate(masks):
                    print(f"  处理目标 {i+1}/{len(masks)}...")

                    # 提取目标的彩色点云
                    pointcloud = self.extract_object_pointcloud(dpt, color_frame, mask)

                    if len(np.asarray(pointcloud.points)) == 0:
                        print(f"    目标 {i+1} 没有有效的点云数据")
                        continue

                    # 对目标进行椭球拟合
                    fitting_result, points_3d = self.fit_ellipsoid_to_object(dpt, color_frame, mask)

                    if fitting_result and fitting_result['success']:
                        print(f"    目标 {i+1} 椭球拟合成功！")
                        print(f"      椭球类型: {fitting_result['type']}")
                        print(f"      内点数量: {fitting_result['num_inliers']}")
                        if fitting_result['center'] is not None:
                            print(f"      椭球中心: {fitting_result['center']}")
                        if fitting_result['axes_lengths'] is not None:
                            print(f"      半轴长度: {fitting_result['axes_lengths']}")

                        # 保存点云
                        name = str(count).zfill(8)
                        ply_filename = base_path + 'pointcloud/' + name + f"_obj_{i+1}" + ".ply"
                        
                        # 确保输出目录存在
                        os.makedirs(os.path.dirname(ply_filename), exist_ok=True)
                        
                        self.converter.save_to_ply(pointcloud, ply_filename)
                        print(f"      点云已保存至: {ply_filename}")
                        
                        # 保存拟合结果用于可视化
                        all_fitting_results.append(fitting_result)
                        all_points_3d.extend(points_3d)  # 收集所有目标的点
                    else:
                        if fitting_result:
                            print(f"    目标 {i+1} 椭球拟合失败: {fitting_result['message']}")
                        else:
                            print(f"    目标 {i+1} 无法提取3D点")

                # 如果有拟合结果，进行可视化
                if all_fitting_results:
                    print(f"  开始可视化 {len(all_fitting_results)} 个椭球拟合结果...")
                    print("  请稍等，生成可视化中...")
                    
                    try:
                        # 可视化所有拟合结果
                        visualize_all_objects_with_colored_pointcloud(
                            dpt, 
                            color_frame, 
                            CAMERA_INTRINSICS, 
                            all_fitting_results,
                            masks
                        )
                    except Exception as e:
                        print(f"  可视化过程中出现错误: {e}")
                        import traceback
                        traceback.print_exc()

                # 检查是否没有检测到任何目标
                if masks is not None and len(masks) == 0:
                    print("未检测到任何目标，对整个点云进行体素下采样和椭球拟合...")
                    
                    # 从深度图和彩色图提取整个场景的点云
                    from pointcloud_utils_new import DepthToPointCloudConverter
                    import open3d as o3d
                    
                    converter = DepthToPointCloudConverter(
                        fx=CAMERA_INTRINSICS["fx"],
                        fy=CAMERA_INTRINSICS["fy"],
                        cx=CAMERA_INTRINSICS["cx"],
                        cy=CAMERA_INTRINSICS["cy"],
                        depth_scale=DEPTH_SCALE
                    )
                    
                    # 转换整个场景的点云（应用距离过滤）
                    full_pointcloud = converter.convert_with_color(
                        dpt, 
                        color_frame, 
                        FILTER_MIN_DIST, 
                        FILTER_MAX_DIST
                    )
                    
                    if len(full_pointcloud.points) > 0:
                        print(f"整个场景点云包含 {len(full_pointcloud.points)} 个点")
                        
                        # 对整个点云进行体素下采样
                        if len(full_pointcloud.points) > 1000:  # 如果点太多，进行下采样
                            print("对整个点云进行体素下采样...")
                            voxel_size = 0.01  # 1cm体素大小
                            downsampled_pcd = full_pointcloud.voxel_down_sample(voxel_size)
                            print(f"下采样后点云包含 {len(downsampled_pcd.points)} 个点")
                        else:
                            downsampled_pcd = full_pointcloud
                            
                        # 将点云转换为numpy数组用于椭球拟合
                        points_for_fitting = np.asarray(downsampled_pcd.points)
                        
                        if len(points_for_fitting) > 0:
                            # 对下采样后的点云进行椭球拟合
                            print("对下采样后的整个点云进行椭球拟合...")
                            from ellipsoid_fitting_improved import fit_ellipsoid_to_points
                            
                            fitting_result = fit_ellipsoid_to_points(
                                points_for_fitting,
                                ransac_threshold=0.02,
                                min_inliers_for_object=50,
                                dbscan_eps=0.05
                            )
                            
                            if fitting_result and fitting_result['success']:
                                print("整个场景椭球拟合成功！")
                                print(f"  椭球类型: {fitting_result['type']}")
                                print(f"  内点数量: {fitting_result['num_inliers']}")
                                if fitting_result['center'] is not None:
                                    print(f"  椭球中心: {fitting_result['center']}")
                                if fitting_result['axes_lengths'] is not None:
                                    print(f"  半轴长度: {fitting_result['axes_lengths']}")
                                
                                # 保存整个场景的点云
                                name = str(count).zfill(8)
                                ply_filename = base_path + 'pointcloud/' + name + "_full_scene" + ".ply"
                                
                                # 确保输出目录存在
                                os.makedirs(os.path.dirname(ply_filename), exist_ok=True)
                                
                                # 保存点云
                                o3d.io.write_point_cloud(ply_filename, downsampled_pcd, write_ascii=True)
                                print(f"  整个场景点云已保存至: {ply_filename}")
                                
                                # 创建一个掩码用于可视化（全图）
                                full_mask = np.ones((dpt.shape[0], dpt.shape[1]), dtype=np.uint8)
                                
                                # 可视化整个场景的拟合结果
                                try:
                                    from advanced_visualization_fixed import visualize_single_object_with_colored_pointcloud
                                    visualize_single_object_with_colored_pointcloud(
                                        dpt, 
                                        color_frame, 
                                        CAMERA_INTRINSICS, 
                                        fitting_result,
                                        mask=full_mask
                                    )
                                except Exception as e:
                                    print(f"  可视化过程中出现错误: {e}")
                            else:
                                if fitting_result:
                                    print(f"整个场景椭球拟合失败: {fitting_result['message']}")
                                else:
                                    print("整个场景椭球拟合失败")
                        else:
                            print("下采样后没有有效点用于椭球拟合")
                    else:
                        print("整个场景没有有效的3D点")

                count += 1
                capture_flag = 0

        # 清理资源
        self.depth_stream.stop()
        self.dev.close()
        cv2.destroyAllWindows()
        self.cap.release()


if __name__ == "__main__":
    try:
        processor = DepthCameraProcessor()
        processor.run()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 确保清理资源
        try:
            openni2.unload()
        except:
            pass