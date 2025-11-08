import numpy as np
import plotly.graph_objects as go
import random


def generate_sample_data(surface_type='sphere', n_points=500, noise=0.05, n_outliers=50):
    """
    生成用于测试的各种二次曲面类型的样本点云数据。
    新增: 可以选择性地添加异常点。
    """
    print(f"正在生成样本数据: {surface_type}...")
    np.random.seed(0)
    x, y, z = None, None, None

    # --- 原始数据点生成 (与原代码相同) ---
    if surface_type == 'sphere':
        r = 5
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
    elif surface_type == 'cylinder':
        r = 3
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-5, 5, n_points)
    elif surface_type == 'cone':
        z_range = np.random.uniform(-5, 5, n_points)
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        r = np.abs(z_range)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z_range
    elif surface_type == 'ellipsoid':
        a, b, c = 5, 3, 2
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        x = a * np.sin(theta) * np.cos(phi)
        y = b * np.sin(theta) * np.sin(phi)
        z = c * np.cos(theta)
    elif surface_type == 'hyperboloid_one_sheet':
        a, b, c = 2, 3, 4
        u = np.random.uniform(0, 2 * np.pi, n_points)
        v = np.random.uniform(-2, 2, n_points)
        x = a * np.cosh(v) * np.cos(u)
        y = b * np.cosh(v) * np.sin(u)
        z = c * np.sinh(v)
    elif surface_type == 'hyperbolic_paraboloid':
        a, b = 2, 2
        u = np.random.uniform(-5, 5, n_points)
        v = np.random.uniform(-5, 5, n_points)
        x = u
        y = v
        z = (u / a) ** 2 - (v / b) ** 2

    if x is not None:
        # 添加噪声
        x += np.random.normal(0, noise, n_points)
        y += np.random.normal(0, noise, n_points)
        z += np.random.normal(0, noise, n_points)

        # --- 新增: 添加异常点 ---
        if n_outliers > 0:
            # 在一个更大的边界框内生成随机点作为异常值
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            z_min, z_max = np.min(z), np.max(z)

            # 扩大边界
            range_x = (x_max - x_min) * 1.5
            range_y = (y_max - y_min) * 1.5
            range_z = (z_max - z_min) * 1.5

            outliers_x = np.random.rand(n_outliers) * range_x + x_min - (range_x - (x_max - x_min)) / 2
            outliers_y = np.random.rand(n_outliers) * range_y + y_min - (range_y - (y_max - y_min)) / 2
            outliers_z = np.random.rand(n_outliers) * range_z + z_min - (range_z - (z_max - z_min)) / 2

            x = np.concatenate([x, outliers_x])
            y = np.concatenate([y, outliers_y])
            z = np.concatenate([z, outliers_z])

    return list(x), list(y), list(z)


def fit_implicit_quadric(x_data, y_data, z_data):
    """
    使用特征值法拟合通用二次曲面隐函数 (与原代码相同)
    Ax^2 + By^2 + Cz^2 + Dxy + Eyz + Fzx + Gx + Hy + Iz + J = 0
    """
    x, y, z = np.array(x_data), np.array(y_data), np.array(z_data)

    if len(x) < 9:
        return None  # 需要至少9个点来确定一个二次曲面

    # 构建设计矩阵 A
    A_matrix = np.vstack([x ** 2, y ** 2, z ** 2, x * y, y * z, z * x, x, y, z, np.ones(len(x))]).T

    # 计算散布矩阵 M = A.T @ A
    M = A_matrix.T @ A_matrix

    # 求解 M 的特征值和特征向量
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # 最小特征值对应的特征向量就是我们的解
        coeffs = eigenvectors[:, 0]
        return tuple(coeffs)
    except np.linalg.LinAlgError:
        return None


def ransac_fit_quadric(x_data, y_data, z_data, min_samples=9, threshold=0.2, max_iterations=500):
    """
    【新增】使用RANSAC算法从可能包含异常点的数据中拟合二次曲面。

    参数:
    - min_samples: 拟合模型所需的最小点数 (二次曲面为9)。
    - threshold: 用于确定一个点是否为内点的距离阈值。
    - max_iterations: RANSAC算法的最大迭代次数。

    返回:
    - best_coeffs: 最佳拟合模型的系数。
    - best_inlier_indices: 最佳内点集的索引。
    """
    print(f"\n开始 RANSAC 拟合... (阈值={threshold}, 迭代次数={max_iterations})")

    x_data, y_data, z_data = np.array(x_data), np.array(y_data), np.array(z_data)
    full_data = np.vstack([x_data, y_data, z_data]).T
    n_points = len(full_data)

    best_coeffs = None
    best_inlier_indices = []
    max_inliers = 0

    for i in range(max_iterations):
        # 1. 随机选择 min_samples 个点
        sample_indices = random.sample(range(n_points), min_samples)
        sample_points = full_data[sample_indices]

        # 2. 用样本点拟合一个临时模型
        temp_coeffs = fit_implicit_quadric(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2])

        if temp_coeffs is None:
            continue

        A, B, C, D, E, F, G, H, I, J = temp_coeffs

        # 3. 计算所有点到这个临时模型的代数距离，并找到内点
        # F(x,y,z) = Ax^2 + ... + J
        all_values = (A * x_data ** 2 + B * y_data ** 2 + C * z_data ** 2 +
                      D * x_data * y_data + E * y_data * z_data + F * z_data * x_data +
                      G * x_data + H * y_data + I * z_data + J)

        # 计算归一化因子，使距离度量更稳定
        # grad_norm = sqrt((2Ax+Dy+Fz+G)^2 + ... )
        # 为了计算效率，通常在RANSAC中使用简化的代数距离 |F(x,y,z)|
        distances = np.abs(all_values)

        current_inlier_indices = np.where(distances < threshold)[0]
        num_inliers = len(current_inlier_indices)

        # 4. 如果当前模型找到了更多的内点，则更新最佳模型
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_indices = current_inlier_indices

            # 打印进度
            print(f"迭代 {i + 1}/{max_iterations}: 找到新的最佳模型，包含 {max_inliers} 个内点。")

            # 提前退出的条件 (可选): 如果内点足够多，可以提前结束
            if num_inliers > n_points * 0.7:
                print("找到了足够多的内点，提前结束迭代。")
                break

    # 5. 使用找到的最佳内点集，进行最终的精确拟合
    if max_inliers > min_samples:
        print(f"\nRANSAC完成。使用 {max_inliers} 个内点进行最终拟合。")
        inlier_points = full_data[best_inlier_indices]
        best_coeffs = fit_implicit_quadric(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2])
    else:
        print("RANSAC 未能找到足够数量的内点来拟合模型。")

    return best_coeffs, best_inlier_indices


def classify_quadric(coeffs, tol=1e-6):
    """
    根据隐函数系数对二次曲面进行分类 (与原代码相同)
    """
    if coeffs is None:
        return "拟合失败 (No Coeffs)"
    A, B, C, D, E, F, G, H, I, J = coeffs
    Q = np.array(
        [[A, D / 2, F / 2, G / 2], [D / 2, B, E / 2, H / 2], [F / 2, E / 2, C, I / 2], [G / 2, H / 2, I / 2, J]])
    Q_3x3 = Q[:3, :3]
    det_Q = np.linalg.det(Q)
    det_Q_3x3 = np.linalg.det(Q_3x3)
    evals_3x3, _ = np.linalg.eig(Q_3x3)
    pos_evals = np.sum(evals_3x3 > tol)
    neg_evals = np.sum(evals_3x3 < -tol)

    if abs(det_Q_3x3) > tol:  # 非退化中心二次曲面
        if pos_evals == 3 or neg_evals == 3:  # 特征值同号
            if np.sign(det_Q) != np.sign(det_Q_3x3):
                return "椭球面 (Ellipsoid)"
            else:
                return "虚椭球面 (Imaginary Ellipsoid)"
        else:  # 特征值异号
            if np.sign(det_Q) != np.sign(det_Q_3x3):
                return "单叶双曲面 (Hyperboloid of One Sheet)"
            else:
                return "双叶双曲面 (Hyperboloid of Two Sheets)"
    elif abs(det_Q_3x3) < tol and abs(det_Q) > tol:  # 抛物面
        if pos_evals > 0 and neg_evals > 0:
            return "双曲抛物面 (Hyperbolic Paraboloid)"
        else:
            return "椭圆抛物面 (Elliptic Paraboloid)"
    # 可以添加更多对退化情况的判断，如锥面、柱面等
    elif abs(det_Q) < tol:
        if pos_evals == 2 and neg_evals == 1 or pos_evals == 1 and neg_evals == 2:
            return "椭圆锥面 (Elliptic Cone)"
        if (pos_evals == 2 or neg_evals == 2) and np.linalg.matrix_rank(Q) == 3:
            return "椭圆柱面 (Elliptic Cylinder)"
        if (pos_evals == 1 and neg_evals == 1) and np.linalg.matrix_rank(Q) == 3:
            return "双曲柱面 (Hyperbolic Cylinder)"

    return "无法识别或退化曲面 (Unidentified or Degenerate)"


def plot_surface_with_ransac(coeffs, x_data, y_data, z_data, inlier_indices, title='RANSAC 隐函数曲面拟合'):
    """
    【新增】可视化RANSAC拟合结果，区分内点和外点。
    """
    if coeffs is None:
        print("系数为空，无法绘图。")
        return

    A, B, C, D, E, F, G, H, I, J = coeffs
    x_all, y_all, z_all = np.array(x_data), np.array(y_data), np.array(z_data)

    # 创建一个布尔掩码来区分内点和外点
    inlier_mask = np.zeros(len(x_all), dtype=bool)
    inlier_mask[inlier_indices] = True

    # 1. 创建内点 (Inliers) 的散点图轨迹
    trace_inliers = go.Scatter3d(
        x=x_all[inlier_mask], y=y_all[inlier_mask], z=z_all[inlier_mask],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.8),
        name=f'内点 ({len(inlier_indices)})'
    )

    # 2. 创建外点 (Outliers) 的散点图轨迹
    trace_outliers = go.Scatter3d(
        x=x_all[~inlier_mask], y=y_all[~inlier_mask], z=z_all[~inlier_mask],
        mode='markers',
        marker=dict(size=3, color='red', opacity=0.8),
        name=f'外点 ({len(x_all) - len(inlier_indices)})'
    )

    # 3. 创建拟合曲面的等值面轨迹
    x_min, x_max = min(x_all), max(x_all)
    y_min, y_max = min(y_all), max(y_all)
    z_min, z_max = min(z_all), max(z_all)
    margin_x = (x_max - x_min) * 0.1;
    margin_y = (y_max - y_min) * 0.1;
    margin_z = (z_max - z_min) * 0.1

    grid_res = 50
    X, Y, Z = np.mgrid[x_min - margin_x:x_max + margin_x:grid_res * 1j,
              y_min - margin_y:y_max + margin_y:grid_res * 1j,
              z_min - margin_z:z_max + margin_z:grid_res * 1j]

    values = (A * X ** 2 + B * Y ** 2 + C * Z ** 2 + D * X * Y + E * Y * Z + F * Z * X + G * X + H * Y + I * Z + J)

    trace_surface = go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=values.flatten(),
        isomin=0, isomax=0,
        surface_count=1,
        colorscale='Viridis',
        opacity=0.6,
        name='拟合曲面',
        showscale=False
    )

    # 组合轨迹并设置布局
    fig = go.Figure(data=[trace_inliers, trace_outliers, trace_surface])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X 轴', yaxis_title='Y 轴', zaxis_title='Z 轴',
                   aspectratio=dict(x=1, y=1, z=1)),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1, traceorder='normal')
    )
    fig.show()


# --- 主程序 ---
if __name__ == '__main__':

    print("请选择要生成的样本数据类型:")
    print("1: 球面 (Sphere)")
    print("2: 圆柱面 (Cylinder)")
    print("3: 椭球面 (Ellipsoid)")
    print("4: 单叶双曲面 (Hyperboloid of One Sheet)")
    print("5: 双曲抛物面 (Hyperbolic Paraboloid)")
    print("6: 圆锥面 (Cone)")

    choice = input("请输入选项 (1-6) [默认为1]: ")
    choices = {'1': 'sphere', '2': 'cylinder', '3': 'ellipsoid',
               '4': 'hyperboloid_one_sheet', '5': 'hyperbolic_paraboloid', '6': 'cone'}
    surface_choice = choices.get(choice, 'sphere')

    outlier_choice = input("是否要添加异常点? (y/n) [默认为y]: ").lower()
    num_outliers = 500 if outlier_choice in ['y', 'yes', ''] else 0

    # 生成数据
    x_data, y_data, z_data = generate_sample_data(
        surface_type=surface_choice,
        n_points=1000,
        noise=0.1,
        n_outliers=num_outliers
    )

    if x_data is not None and len(x_data) > 0:
        # 使用 RANSAC 进行拟合
        # 注意: threshold (阈值) 是一个关键参数，需要根据数据的噪声水平和尺度进行调整
        # 对于此示例，0.2 到 0.5 之间的值通常效果不错
        fitted_coeffs, inlier_indices = ransac_fit_quadric(
            x_data, y_data, z_data,
            min_samples=9,  # 二次曲面需要9个点
            threshold=0.3,  # 内点距离阈值，需要调试
            max_iterations=500  # 迭代次数
        )

        if fitted_coeffs is not None:
            print("\n--- 最终拟合结果 ---")
            coeff_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            for name, val in zip(coeff_names, fitted_coeffs):
                print(f"  {name} = {val:.6f}")

            print("\n开始进行曲面分类...")
            surface_type = classify_quadric(fitted_coeffs)
            print(f"--- 识别的曲面类型: {surface_type} ---")

            print("\n正在生成交互式3D预览...")
            plot_title = f'RANSAC 二次曲面拟合: {surface_type}'
            plot_surface_with_ransac(fitted_coeffs, x_data, y_data, z_data, inlier_indices, title=plot_title)