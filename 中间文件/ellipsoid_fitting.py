"""
椭球拟合算法模块
包含RANSAC椭球拟合算法和相关辅助函数
"""
import numpy as np
import plotly.graph_objects as go
import random
from sklearn.cluster import DBSCAN
import open3d as o3d
from visualization_utils import plot_ellipsoid_fitting_result


def fit_implicit_quadric(points):
    """通过最小二乘法拟合二次曲面"""
    if len(points) < 9: return None
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    A_matrix = np.vstack([x ** 2, y ** 2, z ** 2, x * y, y * z, z * x, x, y, z, np.ones(len(x))]).T
    try:
        _, eigenvectors = np.linalg.eigh(A_matrix.T @ A_matrix)
        return tuple(eigenvectors[:, 0])
    except np.linalg.LinAlgError:
        print("警告: 矩阵计算错误，可能由于点共面或共线。")
        return None


def calculate_residuals(points, coeffs):
    """计算点到二次曲面的代数距离"""
    A, B, C, D, E, F, G, H, I, J = coeffs
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (A * x ** 2 + B * y ** 2 + C * z ** 2 + D * x * y + E * y * z + F * z * x + G * x + H * y + I * z + J)


def get_quadric_center(coeffs):
    """计算中心二次曲面（如椭球面、双曲面）的几何中心点坐标"""
    if coeffs is None: return None
    A, B, C, D, E, F, G, H, I, _ = coeffs
    M = np.array([[A, D / 2, F / 2],
                  [D / 2, B, E / 2],
                  [F / 2, E / 2, C]])
    V = np.array([G, H, I])
    try:
        center = -0.5 * np.linalg.inv(M) @ V
        return center
    except np.linalg.LinAlgError:
        return None


def get_principal_axes(coeffs):
    """计算中心二次曲面的中心、轴方向和轴长"""
    if coeffs is None:
        return None, None, None

    center = get_quadric_center(coeffs)
    if center is None:
        return None, None, None

    A, B, C, D, E, F, G, H, I, J = coeffs
    Q = np.array(
        [[A, D / 2, F / 2, G / 2], [D / 2, B, E / 2, H / 2], [F / 2, E / 2, C, I / 2], [G / 2, H / 2, I / 2, J]])
    Q_3x3 = Q[:3, :3]

    try:
        det_Q = np.linalg.det(Q)
        det_Q_3x3 = np.linalg.det(Q_3x3)
        if abs(det_Q_3x3) < 1e-7:
            return center, None, None

        k = det_Q / det_Q_3x3

        evals, evecs = np.linalg.eigh(Q_3x3)

        with np.errstate(invalid='ignore'):
            axis_lengths = np.sqrt(-k / evals)

        return center, evecs, axis_lengths
    except np.linalg.LinAlgError:
        return center, None, None


def ransac_fit_quadric(points, min_samples=9, threshold=0.2, max_iterations=500):
    """使用RANSAC算法稳健地拟合二次曲面"""
    n_points = len(points)
    if n_points < min_samples:
        print("  > RANSAC 警告: 点数不足以进行拟合。")
        return None, []

    best_inlier_indices = []
    best_rmse = np.inf

    for _ in range(max_iterations):
        try:
            sample_indices = random.sample(range(n_points), min_samples)
        except ValueError:
            print("  > RANSAC 警告: 点数不足以采样。")
            return None, []

        temp_coeffs = fit_implicit_quadric(points[sample_indices])
        if temp_coeffs is None: continue

        residuals = np.abs(calculate_residuals(points, temp_coeffs))
        current_inlier_indices = np.where(residuals < threshold)[0]
        num_inliers = len(current_inlier_indices)

        if num_inliers > len(best_inlier_indices):
            best_inlier_indices = current_inlier_indices
            best_rmse = np.sqrt(np.mean(residuals[current_inlier_indices] ** 2))
        elif num_inliers == len(best_inlier_indices) and num_inliers > 0:
            current_rmse = np.sqrt(np.mean(residuals[current_inlier_indices] ** 2))
            if current_rmse < best_rmse:
                best_inlier_indices = current_inlier_indices
                best_rmse = current_rmse

    if len(best_inlier_indices) > min_samples:
        final_coeffs = fit_implicit_quadric(points[best_inlier_indices])
        if final_coeffs is not None:
            final_residuals = np.abs(calculate_residuals(points, final_coeffs))
            final_inlier_indices = np.where(final_residuals < threshold)[0]
            print(f"  > RANSAC: 找到 {len(final_inlier_indices)} 个内点。")
            return final_coeffs, final_inlier_indices
        else:
            print("  > RANSAC 警告: 最终拟合失败。")

    print(f"  > RANSAC: 未找到足够好的模型 (最多找到 {len(best_inlier_indices)} 个内点)。")
    return None, []


def classify_quadric(coeffs):
    """根据系数对二次曲面进行分类"""
    if coeffs is None: return "拟合失败"
    A, B, C, D, E, F, G, H, I, J = coeffs
    Q = np.array(
        [[A, D / 2, F / 2, G / 2], [D / 2, B, E / 2, H / 2], [F / 2, E / 2, C, I / 2], [G / 2, H / 2, I / 2, J]])
    Q_3x3 = Q[:3, :3]
    try:
        det_Q, det_Q_3x3 = np.linalg.det(Q), np.linalg.det(Q_3x3)
        evals_3x3, _ = np.linalg.eig(Q_3x3)
        pos_evals, neg_evals = np.sum(evals_3x3 > 1e-6), np.sum(evals_3x3 < -1e-6)
    except np.linalg.LinAlgError:
        return "矩阵计算错误"

    if abs(det_Q_3x3) > 1e-6:
        if pos_evals == 3 or neg_evals == 3:
            return "椭球面 (Ellipsoid)" if np.sign(det_Q) != np.sign(det_Q_3x3) else "虚椭球面"
        else:
            return "单叶双曲面" if np.sign(det_Q) > 0 else "双叶双曲面"
    elif abs(det_Q_3x3) < 1e-6 and abs(det_Q) > 1e-6:
        return "椭圆抛物面 (Elliptic Paraboloid)" if pos_evals == 2 or neg_evals == 2 else "双曲抛物面"
    elif abs(det_Q) < 1e-6:
        rank_Q, rank_Q_3x3 = np.linalg.matrix_rank(Q, 1e-6), np.linalg.matrix_rank(Q_3x3, 1e-6)
        if rank_Q_3x3 == 2 and rank_Q == 3: return "椭圆/双曲柱面 (Cylinder)"
    return "无法识别或退化曲面"


def validate_and_refine_inliers(points, inlier_indices, eps, min_samples=10):
    """使用DBSCAN验证RANSAC的内点是否形成一个连通的簇"""
    if len(inlier_indices) < min_samples:
        return np.array([], dtype=int)
    inlier_points = points[inlier_indices]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(inlier_points)
    labels = db.labels_

    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        return np.array([], dtype=int)

    largest_cluster_label = unique_labels[np.argmax(counts)]
    mask = (labels == largest_cluster_label)
    refined_indices = inlier_indices[mask]

    if len(refined_indices) < len(inlier_indices) * 0.5:
        print(f"  > 验证失败: 内点分散在多个簇中。最大簇仅占 {len(refined_indices)}/{len(inlier_indices)}。")
        return np.array([], dtype=int)

    print(f"  > 验证通过: 最大簇包含 {len(refined_indices)}/{len(inlier_indices)} 个内点。")
    return refined_indices


def segment_point_cloud(points, ransac_threshold=0.3, min_inliers_for_object=200, min_points_remaining=100,
                        dbscan_eps=0.5):
    """核心分割函数"""
    print("\n开始点云分割...")
    if points is None or len(points) == 0:
        print("错误：输入的点云为空。")
        return [], np.array([], dtype=int)

    unassigned_mask = np.ones(len(points), dtype=bool)
    detected_objects = []

    iteration = 1
    while np.sum(unassigned_mask) >= min_points_remaining:
        print(f"\n--- 分割迭代 #{iteration} ---")
        current_points_indices = np.where(unassigned_mask)[0]
        remaining_points = points[current_points_indices]
        print(f"剩余点数: {len(remaining_points)}")

        if len(remaining_points) < 9:
            print("剩余点数过少，无法继续拟合。")
            break

        coeffs, inlier_indices_relative = ransac_fit_quadric(remaining_points, threshold=ransac_threshold)

        if coeffs is None or len(inlier_indices_relative) < min_inliers_for_object:
            print("未能找到足够大的候选模型，结束分割。")
            break

        original_indices = current_points_indices[inlier_indices_relative]

        print(f"  > RANSAC 找到 {len(original_indices)} 个候选内点，进行 DBSCAN 验证...")
        refined_inlier_indices = validate_and_refine_inliers(points, original_indices, eps=dbscan_eps)

        if len(refined_inlier_indices) < min_inliers_for_object:
            print(f"  > 验证后的对象尺寸过小 (< {min_inliers_for_object})，拒绝此模型。")
            print(f"  > 将 {len(original_indices)} 个原始 RANSAC 内点标记为已处理以避免重复检测。")
            unassigned_mask[original_indices] = False
            iteration += 1
            continue

        print(f"  > 使用 {len(refined_inlier_indices)} 个验证后的内点重新拟合模型...")
        final_coeffs = fit_implicit_quadric(points[refined_inlier_indices])
        if final_coeffs is None:
            print("  > 警告: 最终模型拟合失败，跳过此对象。")
            unassigned_mask[refined_inlier_indices] = False
            iteration += 1
            continue

        obj_type = classify_quadric(final_coeffs)
        center, axes_vectors, axes_lengths = None, None, None

        if obj_type == "椭圆抛物面 (Elliptic Paraboloid)":
            print(f"  > 检测到椭圆抛物面，分类为椭球面。")
            obj_type = "椭球面 "

        if "椭球面" in obj_type or "双曲面" in obj_type:
            center, axes_vectors, axes_lengths = get_principal_axes(final_coeffs)
            if center is not None:
                print(f"  > 计算得到曲面几何中心: {[f'{c:.3f}' for c in center]}")
                if axes_lengths is not None:
                    print(f"  > 计算得到半轴长度: {[f'{l:.3f}' for l in axes_lengths if not np.isnan(l)]}")
            else:
                print("  > 曲面为非中心曲面或计算失败，无唯一几何中心。")

        detected_objects.append({
            'coeffs': final_coeffs,
            'inlier_indices': refined_inlier_indices,
            'type': obj_type,
            'num_inliers': len(refined_inlier_indices),
            'center': center,
            'axes_vectors': axes_vectors,
            'axes_lengths': axes_lengths
        })
        print(f"成功检测到一个对象: {obj_type}，包含 {len(refined_inlier_indices)} 个内点。")

        unassigned_mask[refined_inlier_indices] = False
        iteration += 1

    noise_indices = np.where(unassigned_mask)[0]
    print(f"\n分割完成。检测到 {len(detected_objects)} 个对象，剩余 {len(noise_indices)} 个噪声点。")
    return detected_objects, noise_indices


def fit_ellipsoid_to_points(points, ransac_threshold=0.01, min_inliers_for_object=50, dbscan_eps=0.05):
    """
    对特定点集进行椭球拟合
    
    Args:
        points: 点云数据 (N, 3)
        ransac_threshold: RANSAC阈值
        min_inliers_for_object: 最小内点数
        dbscan_eps: DBSCAN参数
        
    Returns:
        拟合结果字典
    """
    if len(points) < 9:
        return {'success': False, 'message': '点数不足'}
    
    # 使用RANSAC拟合椭球
    coeffs, inlier_indices = ransac_fit_quadric(
        points, 
        threshold=ransac_threshold, 
        min_samples=9,
        max_iterations=500
    )
    
    if coeffs is None:
        return {'success': False, 'message': '椭球拟合失败'}
    
    # 验证内点
    refined_inlier_indices = validate_and_refine_inliers(
        points, 
        inlier_indices, 
        eps=dbscan_eps,
        min_samples=10
    )
    
    if len(refined_inlier_indices) < min_inliers_for_object:
        return {'success': False, 'message': f'验证后的内点数不足: {len(refined_inlier_indices)}'}
    
    # 重新拟合模型
    final_coeffs = fit_implicit_quadric(points[refined_inlier_indices])
    if final_coeffs is None:
        return {'success': False, 'message': '最终椭球拟合失败'}
    
    obj_type = classify_quadric(final_coeffs)
    
    # 检查是否为椭球面
    if "椭球面" not in obj_type:
        return {'success': False, 'message': f'拟合结果不是椭球面: {obj_type}'}
    
    # 计算椭球的中心和轴
    center, axes_vectors, axes_lengths = get_principal_axes(final_coeffs)
    
    return {
        'success': True,
        'coeffs': final_coeffs,
        'inlier_indices': refined_inlier_indices,
        'type': obj_type,
        'num_inliers': len(refined_inlier_indices),
        'center': center,
        'axes_vectors': axes_vectors,
        'axes_lengths': axes_lengths
    }


def plot_ellipsoid_fitting_result(points, fitting_result):
    """可视化椭球拟合结果"""
    fig_data = []
    
    # 绘制原始点
    if len(points) > 0:
        fig_data.append(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers', 
            marker=dict(size=2, color='gray', opacity=0.6),
            name='原始点云',
        ))
    
    # 如果有拟合结果
    if fitting_result['success']:
        inlier_points = points[fitting_result['inlier_indices']]
        
        # 绘制内点
        fig_data.append(go.Scatter3d(
            x=inlier_points[:, 0], y=inlier_points[:, 1], z=inlier_points[:, 2],
            mode='markers', 
            marker=dict(size=3, color='red'),
            name=f'椭球内点 ({len(inlier_points)} 点)',
        ))
        
        # 如果有中心点
        if fitting_result['center'] is not None:
            center = fitting_result['center']
            fig_data.append(go.Scatter3d(
                x=[center[0]], y=[center[1]], z=[center[2]],
                mode='markers',
                marker=dict(size=8, color='blue', symbol='cross'),
                name='椭球中心'
            ))
    
    fig = go.Figure(data=fig_data)
    fig.update_layout(
        title='椭球拟合结果',
        scene=dict(
            xaxis_title='X轴', 
            yaxis_title='Y轴', 
            zaxis_title='Z轴',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='data'
        ),
        legend_title="图例"
    )
    fig.show()


if __name__ == '__main__':
    # 测试椭球拟合
    print("测试椭球拟合算法...")
    
    # 生成一些测试数据（近似椭球面的点）
    u = np.random.uniform(0, 2*np.pi, 500)
    v = np.random.uniform(0, np.pi, 500)
    a, b, c = 2.0, 1.5, 1.0  # 椭球半轴
    
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    # 添加一些噪声
    noise = 0.05
    x += np.random.normal(0, noise, size=x.shape)
    y += np.random.normal(0, noise, size=y.shape)
    z += np.random.normal(0, noise, size=z.shape)
    
    points = np.column_stack((x, y, z))
    
    print(f"生成了 {len(points)} 个测试点")
    
    # 进行椭球拟合
    result = fit_ellipsoid_to_points(points, 
                                   ransac_threshold=0.1, 
                                   min_inliers_for_object=100, 
                                   dbscan_eps=0.2)
    
    if result['success']:
        print(f"椭球拟合成功！")
        print(f"椭球类型: {result['type']}")
        print(f"内点数量: {result['num_inliers']}")
        if result['center'] is not None:
            print(f"椭球中心: {result['center']}")
        if result['axes_lengths'] is not None:
            print(f"半轴长度: {result['axes_lengths']}")
    else:
        print(f"椭球拟合失败: {result['message']}")
    
    print("椭球拟合测试完成")