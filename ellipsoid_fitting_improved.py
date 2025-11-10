"""
椭球拟合算法模块 - 改进版
包含对部分点云的椭球拟合策略
"""
import numpy as np
import random
from sklearn.cluster import DBSCAN
import math


def fit_implicit_quadric(points):
    """通过最小二乘法拟合二次曲面"""
    if len(points) < 9: return None
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    A_matrix = np.vstack([x ** 2, y ** 2, z ** 2, x * y, y * z, z * x, x, y, z, np.ones(len(x))]).T
    try:
        _, eigenvectors = np.linalg.eigh(A_matrix.T @ A_matrix)
        return tuple(eigenvectors[:, 0])  # 返回最小特征值对应的特征向量
    except np.linalg.LinAlgError:
        print("警告: 矩阵计算错误，可能由于点共面或共线。")
        return None


def calculate_residuals(points, coeffs):
    """计算点到二次曲面的代数距离"""
    if coeffs is None: return np.array([])
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
            axis_lengths = np.sqrt(np.abs(k / evals))  # 使用abs来处理负值情况

        return center, evecs, axis_lengths
    except np.linalg.LinAlgError:
        return center, None, None


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


def ransac_fit_quadric(points, min_samples=9, threshold=0.02, max_iterations=500):
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


def force_ellipsoid_fitting(points, ransac_threshold=0.02, min_inliers_for_object=50, dbscan_eps=0.05):
    """
    强制椭球拟合：对部分点云也尝试拟合为椭球
    主要用于处理只看到椭球一部分的情况
    """
    print(f"  > 尝试强制椭球拟合，点数: {len(points)}")
    
    if len(points) < 9:
        return {'success': False, 'message': '点数不足'}

    # 第一步：尝试标准RANSAC椭球拟合
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

    # 第二步：如果标准拟合不成功或不是椭球，尝试几何拟合方法
    print("  > 标准拟合未成功，尝试几何椭球拟合...")
    return geometric_ellipsoid_fitting(points, min_inliers_for_object)


def geometric_ellipsoid_fitting(points, min_inliers_for_object):
    """
    基于几何的椭球拟合方法，适用于部分点云
    通过计算点云的主轴和边界来估计椭球参数
    """
    if len(points) < 6:  # 至少需要6个点来估计椭球
        return {'success': False, 'message': '点数不足以进行几何拟合'}
    
    # 计算点云的质心
    center = np.mean(points, axis=0)
    
    # 使用PCA计算主轴方向，但保持原始中心
    original_center = center  # 保持原始的点云质心
    centered_points = points - center
    cov_matrix = np.cov(centered_points.T)
    
    try:
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值大小排序（从大到小）
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 估计轴长（改进方法：考虑点云的实际分布范围）
        axes_lengths = np.zeros(3)
        
        for i in range(3):
            # 投影到当前主轴上
            projections = np.dot(centered_points, eigenvectors[:, i])
            
            # 使用更稳健的方法估计轴长
            # 1. 计算投影范围
            proj_min, proj_max = np.min(projections), np.max(projections)
            proj_range = proj_max - proj_min
            
            # 2. 计算95%分位数范围（更稳健）
            lower_95 = np.percentile(projections, 2.5)
            upper_95 = np.percentile(projections, 97.5)
            robust_range = upper_95 - lower_95
            
            # 选择两个估算中较大的，以确保覆盖
            est_length = max(proj_range * 0.6, robust_range * 0.6)  # 转换为半轴长度
            axes_lengths[i] = max(est_length, 0.001)  # 确保不为0
        
        # 确保轴长度为正
        axes_lengths = np.abs(axes_lengths)
        
        # 生成椭球表面的虚拟点用于验证
        # 从估计的椭球参数生成一些点来验证拟合质量
        valid_points = []
        for _ in range(min(100, len(points))):  # 生成验证点
            # 随机生成椭球上的点
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = axes_lengths[0] * np.cos(theta) * np.sin(phi)
            y = axes_lengths[1] * np.sin(theta) * np.sin(phi) 
            z = axes_lengths[2] * np.cos(phi)
            
            # 转换到实际空间
            point_local = np.array([x, y, z])
            point_global = center + np.dot(point_local, eigenvectors.T)
            valid_points.append(point_global)
        
        valid_points = np.array(valid_points)
        
        # 评估拟合质量：计算点到估计椭球的平均距离
        # 首先需要使用更准确的方法来计算距离
        distances = []
        for p in points:
            # 将点转换到椭球局部坐标系
            p_local = np.dot(p - center, eigenvectors)
            # 计算在椭球坐标系中的归一化坐标
            x_norm = p_local[0] / axes_lengths[0]
            y_norm = p_local[1] / axes_lengths[1]
            z_norm = p_local[2] / axes_lengths[2]
            
            # 椭球方程: x² + y² + z² = 1
            # 如果点在椭球上，则 x² + y² + z² = 1
            # 距离可近似为 |sqrt(x² + y² + z²) - 1|
            point_dist = abs(np.sqrt(x_norm**2 + y_norm**2 + z_norm**2) - 1.0)
            distances.append(point_dist)
        
        avg_distance = np.mean(distances)
        
        # 计算距离的标准差，用于评估拟合的一致性
        std_distance = np.std(distances) if len(distances) > 1 else 0
        
        # 使用更灵活的判断标准
        # 使用点云的尺寸来决定阈值
        point_range = np.max(points, axis=0) - np.min(points, axis=0)
        max_range = np.max(point_range)
        distance_threshold = max_range * 0.1  # 10%的范围作为距离阈值
        
        # 检查是否满足拟合标准
        fit_quality_ok = (
            avg_distance < distance_threshold and  # 平均距离合理
            std_distance < distance_threshold and  # 标准差也不太大
            len(points) >= 6  # 至少6个点
        )
        
        if fit_quality_ok:
            print(f"  > 几何椭球拟合成功，平均距离: {avg_distance:.4f}")
            
            # 创建一个近似的二次曲面系数（椭球形式）
            # 这(x-center_x)²/a² + (y-center_y)²/b² + (z-center_z)²/c² = 1
            # 转换为一般形式 Ax² + By² + Cz² + ... = 0
            a, b, c = axes_lengths
            cx, cy, cz = center
            
            # 椭球方程: (x-cx)²/a² + (y-cy)²/b² + (z-cz)²/c² - 1 = 0
            # 展开后: x²/a² + y²/b² + z²/c² - 2*cx*x/a² - 2*cy*y/b² - 2*cz*z/c² + cx²/a² + cy²/b² + cz²/c² - 1 = 0
            A = 1.0 / (a * a)
            B = 1.0 / (b * b) 
            C = 1.0 / (c * c)
            G = -2.0 * cx / (a * a)
            H = -2.0 * cy / (b * b)
            I = -2.0 * cz / (c * c)
            J = (cx*cx)/(a*a) + (cy*cy)/(b*b) + (cz*cz)/(c*c) - 1.0
            
            # 其他系数为0（D=E=F=0，椭球不倾斜）
            coeffs_approx = (A, B, C, 0, 0, 0, G, H, I, J)
            
            print(f"  > 几何椭球拟合成功，平均距离: {avg_distance:.4f}, 标准差: {std_distance:.4f}")
            
            return {
                'success': True,
                'coeffs': coeffs_approx,
                'inlier_indices': np.arange(len(points)),  # 所有点都被认为是内点
                'type': '椭球面 (几何估计)',
                'num_inliers': len(points),
                'center': center,
                'axes_vectors': eigenvectors,
                'axes_lengths': axes_lengths,
                'geometric_fitting': True  # 标记为几何拟合
            }
        else:
            return {'success': False, 'message': f'几何拟合质量不佳，平均距离: {avg_distance:.4f}, 标准差: {std_distance:.4f}, 阈值: {distance_threshold:.4f}'}
            
    except np.linalg.LinAlgError:
        return {'success': False, 'message': 'PCA计算失败'}


def fit_ellipsoid_to_points(points, ransac_threshold=0.01, min_inliers_for_object=50, dbscan_eps=0.05):
    """
    对特定点集进行椭球拟合（改进版：支持部分点云）
    
    Args:
        points: 点云数据 (N, 3)
        ransac_threshold: RANSAC阈值
        min_inliers_for_object: 最小内点数
        dbscan_eps: DBSCAN参数

    Returns:
        拟合结果字典
    """
    if len(points) < 6:  # 减少最低点数要求
        return {'success': False, 'message': '点数不足，至少需要6个点'}

    # 首先尝试标准RANSAC拟合
    result = ransac_fit_quadric(
        points,
        threshold=ransac_threshold,
        min_samples=9,
        max_iterations=500
    )

    if result[0] is not None and len(result[1]) >= min_inliers_for_object:
        coeffs, inlier_indices = result
        
        # 验证内点
        refined_inlier_indices = validate_and_refine_inliers(
            points, inlier_indices, eps=dbscan_eps, min_samples=10
        )

        if len(refined_inlier_indices) >= min_inliers_for_object:
            # 重新拟合模型
            final_coeffs = fit_implicit_quadric(points[refined_inlier_indices])
            if final_coeffs is not None:
                obj_type = classify_quadric(final_coeffs)

                # 如果是椭球面，返回结果
                if "椭球面" in obj_type:
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
                else:
                    print(f"  > 拟合结果不是椭球面: {obj_type}，尝试强制椭球拟合")

    # 如果标准方法失败或不是椭球面，尝试强制椭球拟合
    print("  > 尝试强制椭球拟合...")
    return force_ellipsoid_fitting(points, ransac_threshold, min_inliers_for_object, dbscan_eps)


if __name__ == '__main__':
    print("测试改进的椭球拟合算法...")
    
    # 测试完整椭球
    print("\n1. 测试完整椭球拟合:")
    u = np.random.uniform(0, 2*np.pi, 300)
    v = np.random.uniform(0, np.pi, 300)
    a, b, c = 0.1, 0.08, 0.06
    
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    # 添加噪声
    noise = 0.001
    x += np.random.normal(0, noise, size=x.shape)
    y += np.random.normal(0, noise, size=y.shape)
    z += np.random.normal(0, noise, size=z.shape)
    
    full_ellipsoid_points = np.column_stack((x, y, z))
    
    result_full = fit_ellipsoid_to_points(
        full_ellipsoid_points,
        ransac_threshold=0.02,
        min_inliers_for_object=50,
        dbscan_eps=0.05
    )
    
    if result_full['success']:
        print(f"  完整椭球拟合成功！类型: {result_full['type']}")
        print(f"  中心: {result_full['center']}")
        print(f"  轴长: {result_full['axes_lengths']}")
    else:
        print(f"  完整椭球拟合失败: {result_full['message']}")
    
    # 测试部分椭球（模拟部分可见情况）
    print("\n2. 测试部分椭球拟合:")
    # 只取椭球的前半部分
    mask_half = x >= 0  # 只x>0的部分
    half_ellipsoid_points = full_ellipsoid_points[mask_half]
    print(f"  部分椭球点数: {len(half_ellipsoid_points)}")
    
    result_half = fit_ellipsoid_to_points(
        half_ellipsoid_points,
        ransac_threshold=0.02,
        min_inliers_for_object=30,  # 降低要求
        dbscan_eps=0.05
    )
    
    if result_half['success']:
        print(f"  部分椭球强制拟合成功！类型: {result_half['type']}")
        print(f"  中心: {result_half['center']}")
        print(f"  轴长: {result_half['axes_lengths']}")
    else:
        print(f"  部分椭球强制拟合失败: {result_half['message']}")
    
    print("\n椭球拟合算法测试完成")