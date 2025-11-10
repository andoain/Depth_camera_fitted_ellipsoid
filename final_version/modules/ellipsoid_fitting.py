"""
椭球拟合算法模块
包含椭球拟合算法和相关辅助函数
"""
import numpy as np
import random
from sklearn.cluster import DBSCAN


def fit_implicit_quadric(points):
    """拟合二次曲面"""
    if len(points) < 9: 
        return None
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    A_matrix = np.vstack([x ** 2, y ** 2, z ** 2, x * y, y * z, z * x, x, y, z, np.ones(len(x))]).T
    
    try:
        _, eigenvectors = np.linalg.eigh(A_matrix.T @ A_matrix)
        return tuple(eigenvectors[:, 0])
    except np.linalg.LinAlgError:
        return None


def calculate_residuals(points, coeffs):
    """计算点到二次曲面的代数距离"""
    A, B, C, D, E, F, G, H, I, J = coeffs
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (A * x ** 2 + B * y ** 2 + C * z ** 2 + D * x * y + E * y * z + F * z * x + G * x + H * y + I * z + J)


def get_quadric_center(coeffs):
    """计算二次曲面的中心点"""
    if coeffs is None: 
        return None
    
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
    """计算二次曲面的轴长和方向"""
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
            axis_lengths = np.sqrt(np.abs(-k / evals))

        return center, evecs, axis_lengths
    except np.linalg.LinAlgError:
        return center, None, None


def ransac_fit_quadric(points, min_samples=9, threshold=0.2, max_iterations=500):
    """使用RANSAC拟合二次曲面"""
    n_points = len(points)
    if n_points < min_samples:
        return None, []

    best_inlier_indices = []
    best_rmse = np.inf

    for _ in range(max_iterations):
        try:
            sample_indices = random.sample(range(n_points), min_samples)
        except ValueError:
            return None, []

        temp_coeffs = fit_implicit_quadric(points[sample_indices])
        if temp_coeffs is None: 
            continue

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
            return final_coeffs, final_inlier_indices
        else:
            return None, []

    return None, []


def classify_quadric(coeffs):
    """对二次曲面进行分类"""
    if coeffs is None: 
        return "拟合失败"
    
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
        if rank_Q_3x3 == 2 and rank_Q == 3: 
            return "椭圆/双曲柱面 (Cylinder)"
    return "无法识别或退化曲面"


def validate_and_refine_inliers(points, inlier_indices, eps, min_samples=10):
    """使用DBSCAN验证RANSAC的内点"""
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
        return np.array([], dtype=int)

    return refined_indices


def fit_ellipsoid_to_points(points, ransac_threshold=0.01, min_inliers_for_object=50, dbscan_eps=0.05):
    """
    对点云进行椭球拟合
    
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
        'axes_lengths': axes_lengths,
        'rotation_matrix': axes_vectors
    }


def force_ellipsoid_fitting(points, ransac_threshold=0.02, min_inliers_for_object=50, dbscan_eps=0.05):
    """
    强制椭球拟合：对部分点云也尝试拟合为椭球
    """
    if len(points) < 9:
        return {'success': False, 'message': '点数不足'}

    # 首先尝试标准RANSAC椭球拟合
    coeffs, inlier_indices = ransac_fit_quadric(
        points,
        threshold=ransac_threshold,
        min_samples=9,
        max_iterations=500
    )

    if coeffs is not None and len(inlier_indices) >= min_inliers_for_object:
        final_coeffs = fit_implicit_quadric(points[inlier_indices])
        if final_coeffs is not None:
            obj_type = classify_quadric(final_coeffs)
            
            # 如果是椭球面，直接返回
            if "椭球面" in obj_type:
                refined_inlier_indices = validate_and_refine_inliers(
                    points, inlier_indices, eps=dbscan_eps, min_samples=10
                )
                
                if len(refined_inlier_indices) >= min_inliers_for_object:
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

    # 如果标准拟合不成功或不是椭球，尝试几何拟合方法
    return geometric_ellipsoid_fitting(points, min_inliers_for_object)


def geometric_ellipsoid_fitting(points, min_inliers_for_object):
    """
    基于几何的椭球拟合方法，适用于部分点云
    """
    if len(points) < 6:
        return {'success': False, 'message': '点数不足以进行几何拟合'}
    
    # 计算点云的质心
    center = np.mean(points, axis=0)
    
    # 使用PCA计算主轴方向
    centered_points = points - center
    cov_matrix = np.cov(centered_points.T)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值大小排序（从大到小）
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 估计轴长
        axes_lengths = np.zeros(3)
        
        for i in range(3):
            projections = np.dot(centered_points, eigenvectors[:, i])
            proj_min, proj_max = np.min(projections), np.max(projections)
            proj_range = proj_max - proj_min
            
            lower_95 = np.percentile(projections, 2.5)
            upper_95 = np.percentile(projections, 97.5)
            robust_range = upper_95 - lower_95
            
            est_length = max(proj_range * 0.6, robust_range * 0.6)
            axes_lengths[i] = max(est_length, 0.001)
        
        axes_lengths = np.abs(axes_lengths)
        
        # 评估拟合质量
        distances = []
        for p in points:
            p_local = np.dot(p - center, eigenvectors)
            x_norm = p_local[0] / axes_lengths[0]
            y_norm = p_local[1] / axes_lengths[1]
            z_norm = p_local[2] / axes_lengths[2]
            
            point_dist = abs(np.sqrt(x_norm**2 + y_norm**2 + z_norm**2) - 1.0)
            distances.append(point_dist)
        
        avg_distance = np.mean(distances)
        std_distance = np.std(distances) if len(distances) > 1 else 0
        
        point_range = np.max(points, axis=0) - np.min(points, axis=0)
        max_range = np.max(point_range)
        distance_threshold = max_range * 0.1

        fit_quality_ok = (
            avg_distance < distance_threshold and
            std_distance < distance_threshold and
            len(points) >= 6
        )
        
        if fit_quality_ok:
            # 创建近似的二次曲面系数
            a, b, c = axes_lengths
            cx, cy, cz = center
            
            A = 1.0 / (a * a)
            B = 1.0 / (b * b)
            C = 1.0 / (c * c)
            G = -2.0 * cx / (a * a)
            H = -2.0 * cy / (b * b)
            I = -2.0 * cz / (c * c)
            J = (cx*cx)/(a*a) + (cy*cy)/(b*b) + (cz*cz)/(c*c) - 1.0
            
            coeffs_approx = (A, B, C, 0, 0, 0, G, H, I, J)
            
            return {
                'success': True,
                'coeffs': coeffs_approx,
                'inlier_indices': np.arange(len(points)),
                'type': '椭球面 (几何估计)',
                'num_inliers': len(points),
                'center': center,
                'axes_vectors': eigenvectors,
                'axes_lengths': axes_lengths,
                'geometric_fitting': True
            }
        else:
            return {'success': False, 'message': f'几何拟合质量不佳，平均距离: {avg_distance:.4f}, 标准差: {std_distance:.4f}, 阈值: {distance_threshold:.4f}'}
            
    except np.linalg.LinAlgError:
        return {'success': False, 'message': 'PCA计算失败'}