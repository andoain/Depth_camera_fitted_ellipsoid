"""
OpenNI配置脚本
用于确保Orbbec深度相机的OpenNI库正确配置
"""
import os
import sys

def setup_openni():
    """
    设置OpenNI库路径
    """
    # OpenNI库路径，根据您的描述，路径在C:/Program Files/Orbbec/OpenNI/sdk/libs
    openni_lib_path = r"C:\Program Files\Orbbec\OpenNI\sdk\libs"
    
    # 检查路径是否存在
    if os.path.exists(openni_lib_path):
        print(f"找到OpenNI库路径: {openni_lib_path}")
        
        # 添加库路径到系统路径
        if openni_lib_path not in sys.path:
            sys.path.append(openni_lib_path)
        
        # 添加到环境变量（Windows）
        current_path = os.environ.get('PATH', '')
        if openni_lib_path not in current_path:
            os.environ['PATH'] = current_path + os.pathsep + openni_lib_path
            print("已将OpenNI库路径添加到环境变量")
    else:
        print(f"警告: 找不到OpenNI库路径: {openni_lib_path}")
        print("请确认Orbbec OpenNI SDK已正确安装")
    
    # 尝试导入openni
    try:
        import openni
        print("成功导入OpenNI库")
        return True
    except ImportError as e:
        print(f"导入OpenNI库失败: {e}")
        print("请确保OpenNI Python绑定已正确安装")
        return False


if __name__ == "__main__":
    success = setup_openni()
    if success:
        print("\nOpenNI配置完成！")
    else:
        print("\nOpenNI配置失败，请检查安装。")