"""
YOLO检测工具模块
用于目标检测和分割
"""
import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """YOLOv8目标检测器"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, iou_threshold=0.5):
        """初始化检测器"""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect_objects(self, image, classes=None):
        """检测图像中的目标"""
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )
        return results

    def get_detection_masks(self, image, classes=None):
        """获取目标检测的掩码"""
        results = self.detect_objects(image, classes)

        masks = []
        boxes = []
        class_ids = []

        if results and results[0].masks is not None:
            # 分割模型返回掩码
            for mask, box in zip(results[0].masks.data, results[0].boxes):
                mask_np = mask.cpu().numpy().astype(np.uint8)
                masks.append(mask_np)
                boxes.append(box.xyxy.cpu().numpy()[0])
                class_ids.append(int(box.cls.cpu().numpy()[0]))
        elif results and results[0].boxes is not None:
            # 检测模型返回边界框
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
                masks.append(mask)
                boxes.append([x1, y1, x2, y2])
                class_ids.append(int(box.cls.cpu().numpy()[0]))

        return masks, boxes, class_ids

    def draw_detections(self, image, masks=None, boxes=None, class_ids=None, class_names=None):
        """在图像上绘制检测结果"""
        annotated_img = image.copy()

        if masks is not None:
            for i, mask in enumerate(masks):
                color = tuple(np.random.randint(0, 255, 3).tolist())

                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                colored_mask[mask_resized > 0] = color

                annotated_img = cv2.addWeighted(annotated_img, 1.0, colored_mask, 0.3, 0)

                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    label = f"Class {class_ids[i]}"
                    if class_names and class_ids[i] < len(class_names):
                        label = f"{class_names[class_ids[i]]}"
                    cv2.putText(annotated_img, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_img