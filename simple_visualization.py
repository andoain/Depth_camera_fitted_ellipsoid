"""
简单的可视化函数，用于在主程序中安全地显示椭球拟合结果
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import threading


def plot_ellipsoid_fitting_result_async(points, fitting_result):
    """
    异步显示椭球拟合结果，防止阻塞主程序
    """
    def show_plot():
        try:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制点云
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c='gray', s=1, alpha=0.6, label='Point Cloud')
            
            # 如果拟合成功，绘制椭球中心
            if fitting_result and fitting_result.get('success', False):
                center = fitting_result.get('center')
                if center is not None:
                    ax.scatter(*center, color='red', s=100, marker='x', label='Ellipsoid Center')
                    
                    # 显示椭球参数
                    axes_lengths = fitting_result.get('axes_lengths')
                    if axes_lengths is not None:
                        ax.text(center[0], center[1], center[2], 
                               f'Axes: ({axes_lengths[0]:.3f}, {axes_lengths[1]:.3f}, {axes_lengths[2]:.3f})')
                
                # 添加坐标轴
                ax.quiver(center[0] if center is not None else 0, 
                         center[1] if center is not None else 0, 
                         center[2] if center is not None else 0,
                         0.1, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
                ax.quiver(center[0] if center is not None else 0, 
                         center[1] if center is not None else 0, 
                         center[2] if center is not None else 0,
                         0, 0.1, 0, color='g', arrow_length_ratio=0.1, label='Y')
                ax.quiver(center[0] if center is not None else 0, 
                         center[1] if center is not None else 0, 
                         center[2] if center is not None else 0,
                         0, 0, 0.1, color='b', arrow_length_ratio=0.1, label='Z')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Ellipsoid Fitting Result')
            ax.legend()
            
            plt.show()
        except Exception as e:
            print(f"显示椭球拟合结果时出错: {e}")
    
    # 在单独的线程中显示，避免阻塞主程序
    thread = threading.Thread(target=show_plot)
    thread.daemon = True
    thread.start()


def plot_ellipsoid_fitting_result(points, fitting_result):
    """
    使用简单可视化模块可视化椭球拟合结果（非阻塞）
    """
    plot_ellipsoid_fitting_result_async(points, fitting_result)