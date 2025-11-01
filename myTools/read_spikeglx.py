import re
import numpy as np

def get_exp_dir(root_dir, exp_name):
    exp_dirs = list(root_dir.glob(f"{exp_name}*"))
    if len(exp_dirs) > 1:
        print(f"Warning: Multiple directories found starting with {exp_name}:")
        for d in exp_dirs:
            print(f"  {d}")
        exp_dir = exp_dirs[0]  # 最初のディレクトリを使用
        print(f"Using first directory: {exp_dir}")
    elif len(exp_dirs) == 0:
        raise FileNotFoundError(f"No directory found starting with {exp_name}")
    else:
        exp_dir = exp_dirs[0]
    return exp_dir

def read_spikeglx_meta(meta_file):
    """SpikeGLXのメタファイルを読み込む"""
    meta_dict = {}
    with open(meta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                meta_dict[key] = value
    return meta_dict

def get_geometry(meta_dict):
    try:
        geom_str = meta_dict.get('~snsGeomMap', '')
    except:
        return None
    positions = []
    # (shank:x_pos:y_pos:flag) の形式を探す
    pattern = r'\((\d+):([0-9.]+):([0-9.]+):(\d+)\)'
    matches = re.findall(pattern, geom_str)
    
    for match in matches:
        shank, x, y, flag = match
        positions.append([float(x), float(y)])
    return np.array(positions)

# if geom_str:
#     positions = parse_geometry(geom_str)
    
#     # APチャンネルの数を確認（SYNCチャンネルを除外）
#     num_ap_channels = num_channels - 1  # 最後のチャンネルはSYNC
    
#     # recordingをAPチャンネルのみにスライス（最後のSYNCチャンネルを除外）
#     ap_channel_ids = ap_rec.get_channel_ids()[:num_ap_channels]
#     recording_ap = ap_rec.select_channels(ap_channel_ids)
    
#     # 位置情報がAPチャンネル数と一致することを確認
#     if len(positions) == num_ap_channels:
#         # Probeinterfaceを使用してプローブ情報を設定
#         from probeinterface import Probe, ProbeGroup
        
#         probe = Probe(ndim=2, si_units='um')
#         probe.set_contacts(positions=positions, shapes='square', shape_params={'width': 12})
#         probe.set_device_channel_indices(np.arange(num_ap_channels))
#         probe.create_auto_shape()
        
#         # recordingにプローブ情報を設定
#         recording = recording_ap.set_probe(probe, in_place=False)
        
#         print(f"チャンネル位置情報を設定しました: {positions.shape}")
#         print(f"APチャンネル数: {num_ap_channels}, SYNCチャンネルは除外しました")
#     else:
#         print(f"警告: 位置情報数({len(positions)})とAPチャンネル数({num_ap_channels})が一致しません")
#         recording = recording_ap
# else:
#     print("警告: チャンネル位置情報が見つかりませんでした")
