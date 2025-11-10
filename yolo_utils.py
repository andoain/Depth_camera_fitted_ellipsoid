"""
YOLOv8目标检测和分割工具类
"""
import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8目标检测和分割工具类
    """
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, iou_threshold=0.5):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
    def detect_objects(self, image, classes=None):
        """
        检测图像中的目标
        
        Args:
            image: 输入图像
            classes: 指定要检测的目标类别（可选）
            
        Returns:
            results: YOLO检测结果
        """
        results = self.model(
            image, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )
        return results
        
    def get_detection_masks(self, image, classes=None):
        """
        获取目标检测的掩码
        
        Args:
            image: 输入图像
            classes: 指定要检测的目标类别（可选）
            
        Returns:
            masks: 检测到的掩码列表
            boxes: 检测到的边界框列表
            class_ids: 类别ID列表
        """
        results = self.detect_objects(image, classes)
        
        masks = []
        boxes = []
        class_ids = []
        
        # 检查是否有分割掩码
        if results and results[0].masks is not None:
            # 分割模型返回掩码
            for mask, box in zip(results[0].masks.data, results[0].boxes):
                mask_np = mask.cpu().numpy().astype(np.uint8)  # 转换为numpy数组
                masks.append(mask_np)
                boxes.append(box.xyxy.cpu().numpy()[0])  # 边界框 [x1, y1, x2, y2]
                class_ids.append(int(box.cls.cpu().numpy()[0]))
        else:
            # 检测模型返回边界框
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    # 创建边界框掩码
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 1
                    masks.append(mask)
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(int(box.cls.cpu().numpy()[0]))
        
        return masks, boxes, class_ids
        
    def draw_detections(self, image, masks=None, boxes=None, class_ids=None, class_names=None):
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            masks: 掩码列表
            boxes: 边界框列表
            class_ids: 类别ID列表
            class_names: 类别名称字典
            
        Returns:
            annotated_image: 标注后的图像
        """
        annotated_img = image.copy()
        
        if masks is not None:
            for i, mask in enumerate(masks):
                # 生成随机颜色
                color = tuple(np.random.randint(0, 255, 3).tolist())
                
                # 缩放掩码到图像尺寸
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # 创建彩色掩码
                colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                colored_mask[mask_resized > 0] = color
                
                # 将掩码叠加到图像上
                annotated_img = cv2.addWeighted(annotated_img, 1.0, colored_mask, 0.3, 0)
                
                # 绘制边界框
                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加标签
                    label = f"Class {class_ids[i]}"
                    if class_names and class_ids[i] < len(class_names):
                        label = f"{class_names[class_ids[i]]}"
                    cv2.putText(annotated_img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_img


if __name__ == "__main__":
    # 测试YOLO检测器
    detector = YOLODetector()
    
    # 创建一个测试图像（实际使用时替换为摄像头或图像文件）
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 简单测试
    results = detector.detect_objects(test_img)
    print(f"检测到 {len(results[0].boxes)} 个目标")
    
    print("YOLO工具类测试完成")