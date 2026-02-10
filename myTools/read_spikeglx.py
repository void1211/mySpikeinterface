import re
import numpy as np
from pathlib import Path

def get_exp_path(dir_info):
    root_dir = dir_info["root_dir"]
    name = dir_info["name"]
    ep = dir_info["ep"]
    run = dir_info["run"]
    ng = dir_info["ng"]
    nt = dir_info["nt"]

    exp_name = f"{name}_ep{ep}_{run}"

    dict_path = {
        "root": Path(root_dir) / name,
        "exp": Path(root_dir) / name / exp_name,
        "ap": {
            "root": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" / f"{exp_name}_g{ng}_imec0",
            "meta": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" / f"{exp_name}_g{ng}_imec0" / f"{exp_name}_g{ng}_t{nt}.imec0.ap.meta",
            "bin": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" / f"{exp_name}_g{ng}_imec0" / f"{exp_name}_g{ng}_t{nt}.imec0.ap.bin"
        },
        "lf": {
            "root": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" / f"{exp_name}_g{ng}_imec0",
            "meta": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" / f"{exp_name}_g{ng}_imec0" / f"{exp_name}_g{ng}_t{nt}.imec0.lf.meta",
            "bin": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" / f"{exp_name}_g{ng}_imec0" / f"{exp_name}_g{ng}_t{nt}.imec0.lf.bin"
        },
        "obx": {
            "root": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" ,
            "meta": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" / f"{exp_name}_g{ng}_t{nt}.obx0.obx.meta",
            "bin": Path(root_dir) / name / exp_name / f"{exp_name}_g{ng}" / f"{exp_name}_g{ng}_t{nt}.obx0.obx.bin"
        }
    }
    return dict_path

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
    """SpikeGLXメタファイルからgeometry（位置情報）を取得
    
    Returns
    -------
    positions : np.ndarray
        形状: (n_channels, 2) の配列。各行は [x, y] 位置（μm単位）
    """
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

def get_channel_map(meta_dict):
    """SpikeGLXメタファイルからチャンネルマッピングを取得
    
    ~snsChanMapフィールドがある場合、バイナリファイル内のチャンネルインデックスと
    geometryの位置のマッピングを返す。
    
    Returns
    -------
    channel_map : np.ndarray or None
        バイナリファイル内のチャンネルインデックスに対応するgeometryのインデックス
        存在しない場合はNone
    """
    try:
        chan_map_str = meta_dict.get('~snsChanMap', '')
        if not chan_map_str:
            return None
        
        # ~snsChanMapの形式: (shank:channel_index:flag) のリスト
        # 例: "(0:0:1)(0:1:1)..."
        pattern = r'\((\d+):(\d+):(\d+)\)'
        matches = re.findall(pattern, chan_map_str)
        
        if not matches:
            return None
        
        # バイナリファイル内のチャンネル順序でgeometryのインデックスを取得
        # 各マッチは (shank, channel_index_in_binary, flag) の形式
        # channel_index_in_binaryがバイナリファイル内の順序に対応
        channel_map = []
        for match in matches:
            shank, chan_idx, flag = match
            # geometryのインデックスは、~snsGeomMapの順序に対応
            # ここでは、マッチの順序がgeometryの順序と一致していると仮定
            channel_map.append(int(chan_idx))
        
        return np.array(channel_map)
    except:
        return None

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
