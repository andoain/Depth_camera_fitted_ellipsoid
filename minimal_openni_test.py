"""
最小化OpenNI测试脚本
仅测试OpenNI库的基本初始化
"""
import os
import sys

print("开始测试OpenNI初始化...")

# 设置OpenNI库路径
openni_sdk_path = os.path.join(os.path.dirname(__file__), "openniSDKWin64", "sdk", "libs")
print(f"OpenNI SDK路径: {openni_sdk_path}")

# 确保库路径在环境变量中
current_path = os.environ.get('PATH', '')
if openni_sdk_path not in current_path:
    os.environ['PATH'] = current_path + os.pathsep + openni_sdk_path
    print(f"已添加路径到环境变量")
else:
    print("路径已在环境变量中")

# 也检查OpenNI2的子目录
openni2_path = os.path.join(openni_sdk_path, "OpenNI2")
if os.path.exists(openni2_path):
    if openni2_path not in current_path:
        os.environ['PATH'] += os.pathsep + openni2_path
        print(f"已添加OpenNI2路径到环境变量: {openni2_path}")

try:
    print(f"当前环境PATH长度: {len(os.environ['PATH'])}")
except:
    print("无法打印PATH环境变量（编码问题）")

try:
    print("尝试导入openni...")
    from openni import openni2
    print("成功导入OpenNI库")
    
    # 尝试最简单的初始化
    print("尝试初始化OpenNI...")
    try:
        # 先尝试不带路径的初始化
        openni2.initialize()
        print("OpenNI初始化成功（无路径参数）")
        
        # 然后尝试再次用路径初始化，这可能会导致冲突
        openni2.unload()
        
        # 重新初始化带路径
        openni2.initialize(openni_sdk_path)
        print(f"OpenNI初始化成功（带路径: {openni_sdk_path}）")
        
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        
    # 尝试简单的设备操作
    try:
        print("尝试获取设备列表...")
        dev = openni2.Device.open_any()
        print("成功打开设备")
        dev.close()
        print("成功关闭设备")
    except Exception as e:
        print(f"设备操作失败: {e}")
        import traceback
        traceback.print_exc()
        
    openni2.unload()
    print("OpenNI已卸载")
    
except ImportError as e:
    print(f"导入OpenNI失败: {e}")
    print("请确保已安装openni Python包: pip install openni")
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()