from openni import openni2
import numpy as np
import cv2
from pointcloud_utils import DepthToPointCloudConverter

# ------------------------------------------------------------------
# 1. 相机内参
# ------------------------------------------------------------------
CAMERA_INTRINSICS = {
    "fx": 570.3,
    "fy": 570.3,
    "cx": 320.0,
    "cy": 240.0,
}
DEPTH_SCALE = 1000.0

FILTER_MIN_DIST = 0.55  # (米) 过滤掉小于 0.55m 的点
FILTER_MAX_DIST = 8.0  # (米) 过滤掉大于 8.0m 的点 (相机的标称范围)

# ------------------------------------------------------------------
# 3. 初始化转换器
# ------------------------------------------------------------------
converter = DepthToPointCloudConverter(
    fx=CAMERA_INTRINSICS["fx"],
    fy=CAMERA_INTRINSICS["fy"],
    cx=CAMERA_INTRINSICS["cx"],
    cy=CAMERA_INTRINSICS["cy"],
    depth_scale=DEPTH_SCALE
)

if __name__ == "__main__":
    openni2.initialize()
    dev = openni2.Device.open_any()
    print(dev.get_device_info())
    depth_stream = dev.create_depth_stream()
    dev.set_image_registration_mode(True)
    depth_stream.start()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('depth')

    capture_flag = 0
    base_path = 'dataset/'
    count = 1

    while True:
        frame = depth_stream.read_frame()
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')

        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt = cv2.flip(dpt, 1)

        dpt = dpt.astype(np.uint16)

        dim_gray = cv2.convertScaleAbs(dpt, alpha=0.0425)
        depth_colormap = cv2.applyColorMap(dim_gray, 2)
        cv2.imshow('depth', depth_colormap)

        ret, frame = cap.read()
        cv2.imshow('color', frame)

        key = cv2.waitKey(30)
        if int(key) == ord('q'):
            break
        if int(key) == ord('r'):
            capture_flag = 1

        if capture_flag == 1:
            name = str(count).zfill(8)
            depth_filename = base_path + 'depth/' + name + ".png"
            image_filename = base_path + 'image/' + name + ".jpg"
            ply_filename = base_path + 'pointcloud/' + name + ".ply"

            cv2.imwrite(depth_filename, dpt)
            cv2.imwrite(image_filename, frame)

            # ----------------------------------------------------------
            #    现在传入 min_dist 和 max_dist
            # ----------------------------------------------------------
            print(f"正在处理第 {count} 帧 (过滤范围: {FILTER_MIN_DIST}m 到 {FILTER_MAX_DIST}m)...")

            # 转换器现在返回一个 Open3D PointCloud 对象
            pcd = converter.convert(
                dpt,
                min_dist=FILTER_MIN_DIST,
                max_dist=FILTER_MAX_DIST
            )

            # 检查过滤后是否还有点
            if pcd.has_points():
                # 保存点云 (save_to_ply 现在接受 pcd 对象)
                converter.save_to_ply(pcd, ply_filename)
            else:
                print(f"警告: 帧 {name} 在过滤后没有剩余的点，未保存 .ply 文件。")

            count = count + 1
            capture_flag = 0

    depth_stream.stop()
    dev.close()
    cv2.destroyAllWindows()
    cap.release()