import numpy as np
import pandas as pd
from scipy.io import savemat
from pathlib import Path
import torch

from kilosort.io import load_ops


def convert_to_matlab_files_ks4(res_dir: Path | str, recording):
    """Kilosort4 出力フォルダ内の主要な .npy/.tsv を .mat にまとめて保存する。

    MATLAB 側で後処理する前提のため、本関数では型変換は最小限。
    TSV は MATLAB で扱いやすいように構造化配列（structured array）へ変換して保存する。

    Parameters
    ----------
    res_dir : Path | str
        Kilosort4 の結果フォルダ（`spike_times.npy` などがあるディレクトリ）。
    recording : Recording
        SpikeInterface の Recording オブジェクト。
    """
    res_dir = Path(res_dir)

    amplitudes = safe_load_npy(res_dir / "amplitudes.npy")
    channel_map = safe_load_npy(res_dir / "channel_map.npy")
    channel_positions = safe_load_npy(res_dir / "channel_positions.npy")
    channel_shanks = safe_load_npy(res_dir / "channel_shanks.npy")
    cluster_Amplitude = safe_load_csv(res_dir / "cluster_Amplitude.tsv", sep="\t")
    cluster_ContamPct = safe_load_csv(res_dir / "cluster_ContamPct.tsv", sep="\t")
    cluster_group = safe_load_csv(res_dir / "cluster_group.tsv", sep="\t")
    cluster_KSLabel = safe_load_csv(res_dir / "cluster_KSLabel.tsv", sep="\t")
    kept_spikes = safe_load_npy(res_dir / "kept_spikes.npy")
    ops = load_ops(res_dir / "ops.npy")
    pc_features_ind = safe_load_npy(res_dir / "pc_features_ind.npy")
    pc_features = safe_load_npy(res_dir / "pc_features.npy")
    similar_templates = safe_load_npy(res_dir / "similar_templates.npy")
    spike_clusters = safe_load_npy(res_dir / "spike_clusters.npy")
    spike_detection_templates = safe_load_npy(res_dir / "spike_detection_templates.npy")
    spike_positions = safe_load_npy(res_dir / "spike_positions.npy")
    spike_templates = safe_load_npy(res_dir / "spike_templates.npy")
    spike_times = safe_load_npy(res_dir / "spike_times.npy")
    templates = safe_load_npy(res_dir / "templates.npy")
    templates_ind = safe_load_npy(res_dir / "templates_ind.npy")
    whitening_mat = safe_load_npy(res_dir / "whitening_mat.npy")
    whitening_mat_data = safe_load_npy(res_dir / "whitening_mat_dat.npy")
    whitening_mat_inv = safe_load_npy(res_dir / "whitening_mat_inv.npy")

    cluster_snr = calc_snr(recording, templates)
    results_dict = {
        'amplitudes': amplitudes,
        'spike_times': spike_times,
        'spike_positions': spike_positions,
        'pc_features': pc_features,
        'templates': templates,
        'similar_templates': similar_templates,
        'cluster_KSLabel': cluster_KSLabel,
        'cluster_ContamPct': cluster_ContamPct,
        'cluster_Amplitude': cluster_Amplitude,
        'cluster_snr': cluster_snr,
        'spike_clusters': spike_clusters,
        'kept_spikes': kept_spikes,
        'pc_features_ind': pc_features_ind,
        'similar_templates': similar_templates,
        'spike_templates': spike_templates,
        'spike_positions': spike_positions,
        'spike_templates': spike_templates,
        'whitening_mat': whitening_mat,
        'whitening_mat_data': whitening_mat_data,
        'whitening_mat_inv': whitening_mat_inv,
    }

    savemat(res_dir / 'results.mat', results_dict, long_field_names=True, format='5')
    replaced_ops = replace_none_recursive(ops)
    savemat(res_dir / 'ops.mat', {'ops': replaced_ops}, long_field_names=True)


def convert_to_matlab_files(phy_dir: Path | str, recording):
    """Phy フォルダ内のファイルを使って .mat を作成する関数。"""

    phy_dir = Path(phy_dir)

    amplitudes = safe_load_npy(phy_dir / "amplitudes.npy")
    channel_groups = safe_load_npy(phy_dir / "channel_groups.npy")
    channel_map = safe_load_npy(phy_dir / "channel_map.npy")
    channel_map_si = safe_load_npy(phy_dir / "channel_map_si.npy")
    channel_positions = safe_load_npy(phy_dir / "channel_positions.npy")
    cluster_channel_groups = safe_load_csv(phy_dir / "cluster_channel_group.tsv")
    cluster_group = safe_load_csv(phy_dir / "cluster_group.tsv")
    cluster_si_unit_ids = safe_load_csv(phy_dir / "cluster_si_unit_ids.tsv")
    similar_templates = safe_load_npy(phy_dir / "similar_templates.npy")
    spike_clusters = safe_load_npy(phy_dir / "spike_clusters.npy")
    spike_templates = safe_load_npy(phy_dir / "spike_templates.npy")
    spike_times = safe_load_npy(phy_dir / "spike_times.npy")
    templates = safe_load_npy(phy_dir / "templates.npy")
    templates_ind = safe_load_npy(phy_dir / "template_ind.npy")

    cluster_snr = calc_snr(recording, templates)

    results_dict = {
        'amplitudes': amplitudes,
        'channel_groups': channel_groups,
        'channel_map': channel_map,
        'channel_map_si': channel_map_si,
        'channel_positions': channel_positions,
        'cluster_channel_groups': cluster_channel_groups,
        'cluster_group': cluster_group,
        'cluster_si_unit_ids': cluster_si_unit_ids,
        'similar_templates': similar_templates,
        'spike_clusters': spike_clusters,
        'spike_templates': spike_templates,
        'spike_times': spike_times,
        'templates': templates,
        'templates_ind': templates_ind,
        'cluster_snr': cluster_snr,
    }

    savemat(phy_dir / 'results.mat', results_dict, long_field_names=True, format='5')

def replace_none_recursive(data):
    # 辞書の場合の処理
    if isinstance(data, dict):
        return {
            str(k) if isinstance(k, int) else k: replace_none_recursive(v) 
            for k, v in data.items()
        }
    
    # リストやタプルの場合の処理
    elif isinstance(data, (list, tuple)):
        # リスト内包表記で各要素を再帰的に処理
        return type(data)(replace_none_recursive(item) for item in data)
    
    # PyTorchテンソルの場合の処理（CPUに移動してnumpy配列に変換）
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    
    # numpy配列の場合の処理
    elif isinstance(data, np.ndarray):
        return data
    
    # numpy dtype の場合の処理（文字列に変換）
    elif isinstance(data, np.dtype):
        return str(data)
    
    # 値が None の場合の処理
    elif data is None:
        return 'None'
    
    # その他の値（文字列、数値など）はそのまま返す
    else:
        return data

def calc_snr(recording, templates):
    duration_sec = 1.0
    n_channels_tot = recording.get_num_channels()

    n_unit, n_sample, n_chan = templates.shape
    templates_p2p = np.ptp(templates, axis=1)

    best_channels_idx = np.argmax(templates_p2p, axis=1)

    signal_amplitudes = templates_p2p[np.arange(n_unit), best_channels_idx]

    sampling_rate = recording.get_sampling_frequency()
    n_samples_to_load = int(sampling_rate * duration_sec)

    file_size_bytes = recording.get_memory_size()
    itemsize = recording.get_dtype().itemsize
    total_samples = file_size_bytes // (n_channels_tot * itemsize)

    total_frames = recording.get_num_frames()

    start_frame = total_frames // 2

    data_raw = recording.get_traces(start_frame=start_frame, end_frame=start_frame + n_samples_to_load)

    # 中央値（Median）を引いてオフセットやドリフトの影響を除去
    # abs(x - median(x))
    d = np.abs(data_raw - np.median(data_raw, axis=0))

    mad = np.median(d, axis=0)

    # MADを標準偏差(sigma)相当に変換する定数 0.6745 で割る
    noise_levels = mad / 0.6745

    snr_data = []

    for i in range(n_unit):
        # そのユニットのベストチャンネル（最大振幅チャンネル）のインデックス
        best_ch = best_channels_idx[i]
        
        # 信号強度 (Signal): 既に計算済みの templates_p2p から取得
        sig_val = signal_amplitudes[i]
        
        # ノイズレベル (Noise): ベストチャンネルのノイズを使用
        # recordingのチャンネル順序とtemplatesのチャンネル順序が一致している前提です
        if best_ch < len(noise_levels):
            noise_val = noise_levels[best_ch]
        else:
            noise_val = np.nan
            
        # SNR計算 (ゼロ除算を回避)
        if noise_val > 0:
            snr_val = sig_val / noise_val
        else:
            snr_val = 0
            
        snr_data.append({
            'unit_index': i,          # テンプレート内のインデックス
            'snr': snr_val,           # 計算されたSN比
            'best_channel': best_ch,  # 最もS/Nが良いチャンネル
            'signal_p2p': sig_val,    # 信号の高さ
            'noise_std': noise_val    # ノイズの大きさ
        })

    df_snr = pd.DataFrame(snr_data)

    df_save = df_snr.copy()
    if 'unit_index' in df_save.columns:
        df_save = df_save.rename(columns={'unit_index': 'cluster_id'})

    # # 必須ではありませんが、Phyで見やすくするために小数点以下を丸める
    # if 'snr' in df_save.columns:
    #     df_save['snr'] = df_save['snr'].round(3)
        
    # tsv_filename = res_dir / 'cluster_snr.tsv'
    # df_save.to_csv(tsv_filename, sep='\t', index=False)

    # if 'snr' in df_save.columns:
    #     npy_filename = res_dir / 'cluster_snr.npy'
    #     np.save(npy_filename, df_save['snr'].values)

    cluster_snr = np.column_stack((
        df_save['snr'].values,
        df_save['signal_p2p'].values,
        df_save['noise_std'].values
    ))
    
    return cluster_snr

def safe_load_npy(file_path, allow_pickle=True):
    try:
        return np.load(file_path, allow_pickle=allow_pickle)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])

def safe_load_csv(file_path, sep="\t"):
    try:
        return pd.read_csv(file_path, sep=sep)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
