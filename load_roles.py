import os
from pprint import pformat
from cosyvoice.utils.file_utils import load_wav  # 替换为你的实际导入路径

def format_audio(obj):
    """自定义音频对象显示格式"""
    if isinstance(obj, AudioSegment):  # 假设load_wav返回的是AudioSegment对象
        return f"<AudioSegment duration={len(obj)/1000}s channels={obj.channels}>"
    return obj

def load_role_data(role_dir="role"):
    """加载角色数据并生成结构化字典"""
    preloaded_data = {}
    
    print(f"🔍 开始加载角色数据，扫描目录: {os.path.abspath(role_dir)}")
    
    for role_name in os.listdir(role_dir):
        role_path = os.path.join(role_dir, role_name)
        if not os.path.isdir(role_path):
            continue

        role_data = {}
        valid_role = False

        # 加载默认样本
        default_data = load_sample_pair(role_path, "sample")
        if default_data:
            role_data["default"] = default_data
            valid_role = True

        # 加载编号样本
        index = 1
        while True:
            sample_data = load_sample_pair(role_path, f"sample{index}")
            if not sample_data:
                break
                
            role_data[str(index)] = sample_data
            index += 1
            valid_role = True

        if valid_role:
            preloaded_data[role_name] = role_data
            print(f"角色 {role_name} 加载完成，共 {len(role_data)} 个样本")
        else:
            print(f"❌ 角色 {role_name} 没有有效样本，跳过加载")

    return preloaded_data

def load_sample_pair(role_path, base_name):
    """加载成对的lab和wav文件"""
    lab_path = os.path.join(role_path, f"{base_name}.lab")
    wav_path = os.path.join(role_path, f"{base_name}.wav")
    data = {}
    
    # 加载文本
    if os.path.exists(lab_path):
        try:
            with open(lab_path, "r", encoding="utf-8") as f:
                data["text"] = f.read().strip()
        except Exception as e:
            print(f"  ❌ 加载文本失败 [{base_name}.lab]: {str(e)}")
            return None

    # 加载音频（使用自定义方法）
    if os.path.exists(wav_path):
        try:
            # 假设你的load_wav签名为：def load_wav(path: str, sr: int) -> AudioSegment
            audio = load_wav(wav_path, 16000)  # 根据实际参数调整
            data["wav"] = audio
        except Exception as e:
            print(f"  ❌ 加载音频失败 [{base_name}.wav]: {str(e)}")
            return None

    # 必须同时存在才视为有效样本
    if "text" in data and "wav" in data:
        return data
    return None

if __name__ == "__main__":
    # 加载测试数据（修改为你的实际路径）
    data = load_role_data(role_dir="role")
    print(data["miaoer"]["default"]["wav"])
    