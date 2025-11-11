# 深度相机椭球拟合系统
代码还在修，参数还没精调（小声）
## 项目简介

本项目实现了一个完整的深度相机数据处理系统，包括目标检测、3D点云重建、椭球拟合和可视化功能。系统能够处理深度相机捕获的场景，通过YOLO分割检测目标，将深度数据转换为彩色点云，并对目标进行椭球拟合，最终可视化结果。

## 系统要求

- Python 3.8+ 
- Windows操作系统（支持Orbbec深度相机）
- Orbbec深度相机（Astra系列）

## 依赖库

```bash
pip install opencv-python
pip install open3d
pip install numpy
pip install scikit-learn
pip install ultralytics
pip install openni
```

## 文件结构

```
final_version/
├── main.py                 # 主程序入口
├── modules/                # 功能模块
│   ├── pointcloud_utils.py    # 点云转换
│   ├── depth_processing.py    # 深度处理
│   ├── ellipsoid_fitting.py  # 椭球拟合
│   ├── advanced_visualization.py # 可视化
│   └── yolo_utils.py         # YOLO检测
├── dataset/               # 数据输出目录
│   └── pointcloud/        # PLY点云文件
├── openniSDKWin64/        # OpenNI SDK
├── OpenNI2.dll            # OpenNI库文件
└── TECHNICAL_DOC.md       # 技术文档
```

## 使用方法

### 1. 环境准备
1. 确保Orbbec深度相机已连接到电脑
2. 安装所需的Python依赖包
3. 确保OpenNI SDK文件夹存在于当前目录

### 2. 运行程序
```bash
python main.py
```

### 3. 交互操作
- 运行程序后，将显示深度图和彩色图
- 按 'r' 键处理当前帧，系统会：
  - 使用YOLO检测场景中的目标
  - 提取目标的彩色点云
  - 对每个目标进行椭球拟合
  - 保存点云文件到 `dataset/pointcloud/`
  - 可视化椭球拟合结果
- 按 'q' 键退出程序

### 4. 可视化窗口操作
- 鼠标左键：旋转视角
- 鼠标右键：平移视角  
- 滚轮：缩放视角
- 按 'q' 键退出可视化

## 主要功能

1. **实时目标检测**：使用YOLOv8进行实时目标检测和分割
2. **彩色点云重建**：从深度图生成彩色3D点云
3. **椭球拟合**：对检测目标进行椭球形状拟合
4. **3D可视化**：显示点云和椭球拟合结果
5. **数据保存**：将点云保存为PLY格式文件
6. **坐标系统一**：确保所有坐标系统一处理

## 输出说明

- **PLY点云文件**：保存在 `dataset/pointcloud/` 目录
  - 命名格式：`XXXXXXXX_obj_Y.ply` (目标点云) 或 `XXXXXXXX_full_scene.ply` (场景点云)
- **拟合结果显示**：在可视化窗口中显示椭球线框和主轴方向

## 注意事项

1. 确保深度相机正确连接且驱动已安装
2. 首次运行YOLO可能需要下载模型文件
3. 部分情况下（如目标只有一侧可见），系统会进行几何拟合
4. 拟合精度取决于点云质量和目标形状完整性

## 应用场景

- 机器人抓取目标识别
- 工业检测中的物体分析
- 3D建模和重建
- 深度传感器数据处理



## 参考文献

[1] M. Han, P. Wang, X. Wang, et al., "Mobile Grabbing Robot with Target Detection Using Quadric Fitting," in *2021 IEEE International Conference on Integrated Circuits, Technologies and Applications (ICTA)*. IEEE, 2021, pp. 199–200.

[2] M. Han, J. Kan, G. Yang, et al., "Robust Ellipsoid Fitting Using Combination of Axial and Sampson Distances," *IEEE Transactions on Instrumentation and Measurement*, vol. 72, pp. 1–14, 2023.

[3] Kelejun, "Astro Pro 1 *Kelejun's Blog*, 2024. [Online]. Available: https://blog.kelejun.cn/article/astropro1/

### readme由qwen协助编写