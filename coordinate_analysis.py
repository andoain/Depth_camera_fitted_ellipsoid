"""
坐标系对齐问题诊断
"""
import numpy as np
from pointcloud_utils_new import COORD_TRANSFORM


def analyze_coordinate_transform():
    """分析坐标变换的影响"""
    print("坐标变换矩阵分析:")
    print(f"变换矩阵:\n{COORD_TRANSFORM}")
    
    # 测试一个点
    original_point = np.array([0.1, 0.05, 0.8])
    print(f"原始点: {original_point}")
    
    # 应用变换
    homog_point = np.array([original_point[0], original_point[1], original_point[2], 1])
    transformed_point = (COORD_TRANSFORM @ homog_point)[:3]
    print(f"变换后点: {transformed_point}")
    
    # 验证变换性质
    print(f"变换是否保持距离不变: {np.allclose(np.linalg.norm(original_point), np.linalg.norm(transformed_point[:3]))}")


def check_ellipsoid_fitting_coordinate_system():
    """检查椭球拟合坐标系统"""
    print("\n椭球拟合坐标系统检查:")
    
    # 假设我们在原始坐标系中有一个椭球中心
    original_center = np.array([0.2, 0.1, 0.8])  # 在原始相机坐标系中
    print(f"原始椭球中心: {original_center}")
    
    # 假设点云经过变换后，椭球拟合得到的结果
    transformed_center = np.array([0.2, -0.1, -0.8])  # 与上面相同位置，但坐标系不同
    print(f"变换后椭球中心: {transformed_center}")
    
    # 问题：如果椭球是在变换后的点云上拟合的，那它的结果已经是在新坐标系中
    # 所以直接用于可视化应该没问题
    print("关键点: 椭球拟合算法输出的中心和轴已经是在统一坐标系中")
    
    
def review_flow():
    """检查整个数据流"""
    print("\n数据流检查:")
    print("1. 深度图 → [原始相机坐标系点云] → 应用坐标变换 → [统一坐标系点云]")
    print("2. 统一点云 → [椭球拟合] → [中心、轴长、旋转(统一坐标系)] → [可视化]")
    print("3. 问题: 拟合结果应该直接在统一坐标系中使用，不需要额外变换")
    

def check_visualization_module():
    """检查可视化模块是否有多余变换"""
    print("\n可视化模块检查:")
    print("- 可视化模块接收的椭球参数已经在统一坐标系中")
    print("- 不应该再应用任何坐标变换")
    print("- 椭球中心、线框、主轴都应该直接使用拟合结果")


if __name__ == "__main__":
    analyze_coordinate_transform()
    check_ellipsoid_fitting_coordinate_system()
    review_flow()
    check_visualization_module()
    
    print("\n可能的问题:")
    print("1. 可视化时对椭球中心或线框应用了额外变换")
    print("2. 拟合算法可能在不同坐标系中进行")
    print("3. 坐标变换的处理顺序有问题")