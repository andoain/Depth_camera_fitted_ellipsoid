    def create_ellipsoid_wireframe(self, center, axes_lengths, rotation_matrix=None, color=[1, 0.7, 0], line_width=2.0):
        """
        创建椭球的线框骨架，避免遮挡点云
        """
        # 创建参数化的椭球表面点
        u = np.linspace(0, 2 * np.pi, 20)  # 经度（减少点数以提高性能）
        v = np.linspace(0, np.pi, 10)      # 纬度（减少点数以提高性能）
        
        all_points = []
        all_lines = []
        line_colors = []
        
        # 经线（纵向线）
        for i in range(len(u)):
            longitude_points = []
            for j in range(len(v)):
                # 标准椭球参数方程
                x = axes_lengths[0] * np.cos(u[i]) * np.sin(v[j])
                y = axes_lengths[1] * np.sin(u[i]) * np.sin(v[j])
                z = axes_lengths[2] * np.cos(v[j])
                longitude_points.append([x, y, z])
            
            # 转换到全局坐标系
            longitude_points = np.array(longitude_points)
            longitude_points += center
            if rotation_matrix is not None:
                longitude_points = longitude_points @ rotation_matrix.T

            # 应用坐标变换
            # 创建齐次坐标
            longitude_homogeneous = np.hstack([longitude_points, np.ones((len(longitude_points), 1))])
            # 应用坐标变换
            longitude_transformed_homogeneous = longitude_homogeneous @ COORD_TRANSFORM.T
            # 转换回3D坐标
            longitude_points = longitude_transformed_homogeneous[:, :3]

            all_points.extend(longitude_points)
            
            # 创建线段
            for k in range(len(longitude_points)-1):
                start_idx = len(all_points) - len(longitude_points) + k
                end_idx = start_idx + 1
                all_lines.append([start_idx, end_idx])
                line_colors.append(color)
        
        # 纬线（横向线）
        for j in range(len(v)):
            latitude_points = []
            for i in range(len(u)):
                # 标准椭球参数方程
                x = axes_lengths[0] * np.cos(u[i]) * np.sin(v[j])
                y = axes_lengths[1] * np.sin(u[i]) * np.sin(v[j])
                z = axes_lengths[2] * np.cos(v[j])
                latitude_points.append([x, y, z])
            
            # 转换到全局坐标系
            latitude_points = np.array(latitude_points)
            latitude_points += center
            if rotation_matrix is not None:
                latitude_points = latitude_points @ rotation_matrix.T

            # 应用坐标变换
            # 创建齐次坐标
            latitude_homogeneous = np.hstack([latitude_points, np.ones((len(latitude_points), 1))])
            # 应用坐标变换
            latitude_transformed_homogeneous = latitude_homogeneous @ COORD_TRANSFORM.T
            # 转换回3D坐标
            latitude_points = latitude_transformed_homogeneous[:, :3]

            all_points.extend(latitude_points)
            
            # 创建线段（连接纬线上的点）
            for k in range(len(latitude_points)-1):
                start_idx = len(all_points) - len(latitude_points) + k
                end_idx = start_idx + 1
                all_lines.append([start_idx, end_idx])
                line_colors.append(color)
            
            # 连接最后一个点到第一个点形成环
            start_idx = len(all_points) - len(latitude_points)
            end_idx = len(all_points) - 1
            all_lines.append([end_idx, start_idx])  # 连接最后一个点到第一个点
            line_colors.append(color)
        
        # 创建LineSet
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(all_points)
        lineset.lines = o3d.utility.Vector2iVector(all_lines)
        lineset.colors = o3d.utility.Vector3dVector(line_colors)
        
        return lineset