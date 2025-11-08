import numpy as np
import plotly.graph_objects as go
import random
from sklearn.cluster import DBSCAN


# ==============================================================================
# 1. 数据生成
# ==============================================================================
def generate_single_object_data(surface_type='sphere', n_points=500, noise=0.05, center=[0, 0, 0]):
    """生成单个二次曲面对象的数据点。"""
    np.random.seed(random.randint(0, 10000))
    x, y, z = None, None, None
    if surface_type == 'sphere':
        r = 4
        phi, costheta = np.random.uniform(0, 2 * np.pi, n_points), np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)
    elif surface_type == 'cylinder':
        r, h = 2.5, 8
        theta, z_vals = np.random.uniform(0, 2 * np.pi, n_points), np.random.uniform(-h / 2, h / 2, n_points)
        x, y, z = r * np.cos(theta), r * np.sin(theta), z_vals
    elif surface_type == 'ellipsoid':
        a, b, c = 5, 3, 2
        phi, costheta = np.random.uniform(0, 2 * np.pi, n_points), np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        x, y, z = a * np.sin(theta) * np.cos(phi), b * np.sin(theta) * np.sin(phi), c * np.cos(theta)

    if x is not None:
        points = np.vstack((x, y, z)).T
        points += np.random.normal(0, noise, points.shape)
        points += np.array(center)
        return points
    return np.empty((0, 3))


def generate_multi_object_data(shapes, n_outliers=500, workspace_size=30):
    """
    【已修正】生成包含多个对象和全局噪声的复杂场景。
    """
    print("正在生成多对象场景...")
    all_points = []
    for shape_info in shapes:
        # --- 错误修正处 ---
        # 复制字典，并将'type'键重命名为'surface_type'以匹配函数签名
        kwargs = shape_info.copy()
        kwargs['surface_type'] = kwargs.pop('type')

        points = generate_single_object_data(**kwargs)
        if points is not None and len(points) > 0:
            all_points.append(points)

    scene_points = np.vstack(all_points)
    outliers = (np.random.rand(n_outliers, 3) - 0.5) * (2 * workspace_size)
    scene_points = np.vstack([scene_points, outliers])
    np.random.shuffle(scene_points)

    print(f"场景创建完毕，共 {len(scene_points)} 个点。")
    return scene_points[:, 0], scene_points[:, 1], scene_points[:, 2]


# ==============================================================================
# 2. 核心拟合与分类函数
# ==============================================================================

def fit_implicit_quadric(points):
    if len(points) < 9: return None
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


def ransac_fit_quadric(points, min_samples=9, threshold=0.2, max_iterations=500):
    n_points = len(points)
    best_coeffs = None
    best_inlier_indices = []
    best_rmse = np.inf

    for _ in range(max_iterations):
        sample_indices = random.sample(range(n_points), min_samples)
        temp_coeffs = fit_implicit_quadric(points[sample_indices])
        if temp_coeffs is None: continue

        residuals = np.abs(calculate_residuals(points, temp_coeffs))
        current_inlier_indices = np.where(residuals < threshold)[0]
        num_inliers = len(current_inlier_indices)

        if num_inliers > len(best_inlier_indices):
            best_inlier_indices = current_inlier_indices
            best_rmse = np.sqrt(np.mean(residuals[current_inlier_indices] ** 2))
        elif num_inliers == len(best_inlier_indices):
            current_rmse = np.sqrt(np.mean(residuals[current_inlier_indices] ** 2))
            if current_rmse < best_rmse:
                best_inlier_indices = current_inlier_indices
                best_rmse = current_rmse

    if len(best_inlier_indices) > min_samples:
        final_coeffs = fit_implicit_quadric(points[best_inlier_indices])
        return final_coeffs, best_inlier_indices

    return None, []


def classify_quadric(coeffs):
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
            return "单叶双曲面" if np.sign(det_Q) != np.sign(det_Q_3x3) else "双叶双曲面"
    elif abs(det_Q_3x3) < 1e-6 and abs(det_Q) > 1e-6:
        return "双曲抛物面" if pos_evals > 0 and neg_evals > 0 else "椭圆抛物面"
    elif abs(det_Q) < 1e-6:
        rank_Q, rank_Q_3x3 = np.linalg.matrix_rank(Q, 1e-6), np.linalg.matrix_rank(Q_3x3, 1e-6)
        if rank_Q_3x3 == 2 and rank_Q == 3: return "椭圆/双曲柱面 (Cylinder)"
    return "无法识别或退化曲面"


# ==============================================================================
# 3. 分割与可视化
# ==============================================================================

def validate_and_refine_inliers(points, inlier_indices, eps, min_samples=10):
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

    print(f"  > 验证成功: 最大簇包含 {len(refined_indices)}/{len(inlier_indices)} 个内点。")
    return refined_indices


def segment_point_cloud(points, ransac_threshold=0.3, min_inliers_for_object=200, min_points_remaining=100,
                        dbscan_eps=0.5):
    print("\n开始点云分割...")
    unassigned_mask = np.ones(len(points), dtype=bool)
    detected_objects = []

    iteration = 1
    while np.sum(unassigned_mask) > min_points_remaining:
        print(f"\n--- 分割迭代 #{iteration} ---")
        remaining_points = points[unassigned_mask]
        print(f"剩余点数: {len(remaining_points)}")

        coeffs, inlier_indices_relative = ransac_fit_quadric(remaining_points, threshold=ransac_threshold)

        if coeffs is None or len(inlier_indices_relative) < min_inliers_for_object:
            print("未能找到足够大的候选模型，结束分割。")
            break

        original_indices = np.where(unassigned_mask)[0][inlier_indices_relative]
        refined_inlier_indices = validate_and_refine_inliers(points, original_indices, eps=dbscan_eps)

        if len(refined_inlier_indices) < min_inliers_for_object:
            print("验证后的对象尺寸过小，拒绝此模型。")
            # 为避免在同一组跨对象的点上反复失败，我们将原始的较大内点集标记为已处理
            unassigned_mask[original_indices] = False
            iteration += 1
            continue

        final_coeffs = fit_implicit_quadric(points[refined_inlier_indices])
        if final_coeffs is None: continue

        obj_type = classify_quadric(final_coeffs)
        detected_objects.append({
            'coeffs': final_coeffs, 'inlier_indices': refined_inlier_indices,
            'type': obj_type, 'num_inliers': len(refined_inlier_indices)
        })
        print(f"成功检测到一个对象: {obj_type}，包含 {len(refined_inlier_indices)} 个内点。")

        unassigned_mask[refined_inlier_indices] = False
        iteration += 1

    noise_indices = np.where(unassigned_mask)[0]
    print(f"\n分割完成。检测到 {len(detected_objects)} 个对象，剩余 {len(noise_indices)} 个噪声点。")
    return detected_objects, noise_indices


def plot_segmentation_results(points, detected_objects, noise_indices):
    fig_data = []
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for i, obj in enumerate(detected_objects):
        color = colors[i % len(colors)]
        inlier_points = points[obj['inlier_indices']]
        fig_data.append(go.Scatter3d(
            x=inlier_points[:, 0], y=inlier_points[:, 1], z=inlier_points[:, 2],
            mode='markers', marker=dict(size=3, color=color),
            name=f"对象 #{i + 1}: {obj['type']} ({obj['num_inliers']} 点)"
        ))
        A, B, C, D, E, F, G, H, I, J = obj['coeffs']
        xmin, xmax = np.min(inlier_points[:, 0]), np.max(inlier_points[:, 0])
        ymin, ymax = np.min(inlier_points[:, 1]), np.max(inlier_points[:, 1])
        zmin, zmax = np.min(inlier_points[:, 2]), np.max(inlier_points[:, 2])
        X, Y, Z = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
        values = (A * X ** 2 + B * Y ** 2 + C * Z ** 2 + D * X * Y + E * Y * Z + F * Z * X + G * X + H * Y + I * Z + J)
        fig_data.append(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=values.flatten(),
            isomin=0, isomax=0, surface_count=1, colorscale=[[0, color], [1, color]],
            opacity=0.3, showscale=False, name=f"曲面 #{i + 1}"
        ))
    noise_points = points[noise_indices]
    fig_data.append(go.Scatter3d(
        x=noise_points[:, 0], y=noise_points[:, 1], z=noise_points[:, 2],
        mode='markers', marker=dict(size=2, color='gray', opacity=0.5),
        name=f"噪声 ({len(noise_points)} 点)"
    ))
    fig = go.Figure(data=fig_data)
    fig.update_layout(title='点云分割结果', scene=dict(aspectratio=dict(x=1, y=1, z=1)))
    fig.show()


# ==============================================================================
# 4. 主程序
# ==============================================================================
if __name__ == '__main__':
    scene_description = [
        {'type': 'ellipsoid', 'n_points': 1000, 'noise': 0.1, 'center': [0, 0, 0]},
        {'type': 'sphere', 'n_points': 1000, 'noise': 0.1, 'center': [10, 0, 0]},
        {'type': 'cylinder', 'n_points': 800, 'noise': 0.15, 'center': [0, 15, 0]},
    ]

    x_data, y_data, z_data = generate_multi_object_data(
        shapes=scene_description, n_outliers=800, workspace_size=25
    )
    all_points = np.vstack([x_data, y_data, z_data]).T

    detected_objects, noise_indices = segment_point_cloud(
        all_points,
        ransac_threshold=0.25,
        min_inliers_for_object=400,
        min_points_remaining=300,
        dbscan_eps=1
    )

    if detected_objects:
        plot_segmentation_results(all_points, detected_objects, noise_indices)
    else:
        print("没有检测到任何对象。")