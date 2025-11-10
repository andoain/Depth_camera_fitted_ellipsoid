"""
OpenNI连接测试脚本
用于单独测试OpenNI库和设备连接
"""
import os
import sys

# 设置OpenNI库路径
openni_sdk_path = os.path.join(os.path.dirname(__file__), "openniSDKWin64", "sdk", "libs")
if os.path.exists(openni_sdk_path):
    # 添加库路径到环境变量（Windows）
    current_path = os.environ.get('PATH', '')
    if openni_sdk_path not in current_path:
        os.environ['PATH'] = current_path + os.pathsep + openni_sdk_path
    print(f"已将OpenNI库路径添加到环境变量: {openni_sdk_path}")
else:
    print(f"警告: 找不到OpenNI SDK路径: {openni_sdk_path}")

try:
    from openni import openni2
    print("成功导入OpenNI库")
    
    print(f"尝试初始化OpenNI，路径: {openni_sdk_path}")
    if os.path.exists(openni_sdk_path):
        openni2.initialize(openni_sdk_path)
        print("OpenNI初始化成功")
    else:
        openni2.initialize()
        print("OpenNI初始化成功（使用默认路径）")
    
    # 尝试打开设备 - 这是检测设备是否可用的方法
    print("尝试打开设备...")
    try:
        dev = openni2.Device.open_any()
        print("设备打开成功")
        print("设备信息:")
        print(dev.get_device_info())
        
        # 尝试创建深度流
        print("尝试创建深度流...")
        depth_stream = dev.create_depth_stream()
        print("深度流创建成功")
        
        # 尝试读取一帧
        print("尝试读取一帧数据...")
        frame = depth_stream.read_frame()
        print(f"成功读取帧，宽度: {frame.width}, 高度: {frame.height}")
        
        # 清理
        depth_stream.stop()
        depth_stream.destroy()
        dev.close()
        print("测试完成，设备已关闭")
        
    except Exception as e:
        print(f"设备操作失败: {e}")
        import traceback
        traceback.print_exc()
        
    openni2.unload()
    print("OpenNI已卸载")
    
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()