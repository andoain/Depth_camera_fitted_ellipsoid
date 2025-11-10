"""
椭球拟合可视化和机器人抓取参数提取示例
"""
import numpy as np
from ellipsoid_fitting_new import fit_ellipsoid_to_points, plot_ellipsoid_fitting_result
from visualization_utils import get_ellipsoid_parameters_for_grasping, PointCloudVisualizer


def create_test_ellipsoid_points(a=0.1, b=0.08, c=0.06, n_points=500):
    """创建测试椭球点云"""
    # 生成椭球参数化点
    u = np.random.uniform(0, 2*np.pi, n_points)
    v = np.random.uniform(0, np.pi, n_points)
    
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    # 添加噪声
    noise = 0.002
    x += np.random.normal(0, noise, size=x.shape)
    y += np.random.normal(0, noise, size=y.shape)
    z += np.random.normal(0, noise, size=z.shape)
    
    # 添加一些随机点作为噪声
    n_noise = 50
    x_noise = np.random.uniform(-a*1.5, a*1.5, n_noise)
    y_noise = np.random.uniform(-b*1.5, b*1.5, n_noise)
    z_noise = np.random.uniform(-c*1.5, c*1.5, n_noise)
    
    x = np.concatenate([x, x_noise])
    y = np.concatenate([y, y_noise])
    z = np.concatenate([z, z_noise])
    
    points = np.column_stack((x, y, z))
    return points


def main():
    print("椭球拟合可视化和机器人抓取参数提取示例")
    print("=" * 50)
    
    # 创建测试点云
    print("生成测试椭球点云...")
    test_points = create_test_ellipsoid_points(a=0.1, b=0.08, c=0.06)
    print(f"生成了 {len(test_points)} 个点")
    
    # 椭球拟合
    print("\n进行椭球拟合...")
    result = fit_ellipsoid_to_points(
        test_points,
        ransac_threshold=0.02,
        min_inliers_for_object=100,
        dbscan_eps=0.05
    )
    
    if result['success']:
        print("椭球拟合成功！")
        print(f"  - 椭球类型: {result['type']}")
        print(f"  - 内点数量: {result['num_inliers']}")
        if result['center'] is not None:
            print(f"  - 椭球中心: [{result['center'][0]:.3f}, {result['center'][1]:.3f}, {result['center'][2]:.3f}]")
        if result['axes_lengths'] is not None:
            print(f"  - 半轴长度: [{result['axes_lengths'][0]:.3f}, {result['axes_lengths'][1]:.3f}, {result['axes_lengths'][2]:.3f}]")
    else:
        print(f"椭球拟合失败: {result['message']}")
        return
    
    # 可视化结果
    print("\n可视化椭球拟合结果...")
    plot_ellipsoid_fitting_result(test_points, result)
    
    # 提取机器人抓取参数
    print("\n提取机器人抓取参数...")
    grasp_params = get_ellipsoid_parameters_for_grasping(result)
    
    if grasp_params:
        print("提取的机器人抓取参数:")
        print(f"  - 物体中心位置: [{grasp_params['center'][0]:.3f}, {grasp_params['center'][1]:.3f}, {grasp_params['center'][2]:.3f}]")
        print(f"  - 三轴长度: [{grasp_params['axes_lengths'][0]:.3f}, {grasp_params['axes_lengths'][1]:.3f}, {grasp_params['axes_lengths'][2]:.3f}]")
        print(f"  - 体积估算: {grasp_params['volume']:.6f} m3")
        print(f"  - 物体尺寸分类: {grasp_params['size_category']}")
        print(f"  - 主轴方向: [{grasp_params['principal_axis'][0]:.3f}, {grasp_params['principal_axis'][1]:.3f}, {grasp_params['principal_axis'][2]:.3f}]")
        
        # 为机器人抓取计算姿态
        visualizer = PointCloudVisualizer()
        grasp_pose = visualizer.visualize_robot_grasp_pose(result, gripper_offset=0.02)
        if grasp_pose:
            print(f"\n机器人抓取姿态:")
            print(f"  - 抓取位置: [{grasp_pose['grasp_position'][0]:.3f}, {grasp_pose['grasp_position'][1]:.3f}, {grasp_pose['grasp_position'][2]:.3f}]")
            print(f"  - 接近方向: [{grasp_pose['approach_vector'][0]:.3f}, {grasp_pose['approach_vector'][1]:.3f}, {grasp_pose['approach_vector'][2]:.3f}]")
        
        print("\n可视化完成！请查看3D图形窗口。")
    else:
        print("无法提取机器人抓取参数。")


if __name__ == "__main__":
    main()