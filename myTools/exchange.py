import numpy as np
import pandas as pd
from scipy.io import savemat
from pathlib import Path

from kilosort.io import load_ops

def exchange_kilosort4_results(res_dir: Path | str, save_path: Path | str):
    """Kilosort4 出力フォルダ内の主要な .npy/.tsv を .mat にまとめて保存する。

    MATLAB 側で後処理する前提のため、本関数では型変換は最小限。
    TSV は MATLAB で扱いやすいように構造化配列（structured array）へ変換して保存する。

    Parameters
    ----------
    res_dir : Path | str
        Kilosort4 の結果フォルダ（`spike_times.npy` などがあるディレクトリ）。
    save_path : Path | str
        出力する .mat ファイルのパス。
    """
    res_dir = Path(res_dir)
    save_path = Path(save_path)
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


    data = {
        "amplitudes": amplitudes,
        "channel_map": channel_map,
        "channel_positions": channel_positions,
        "channel_shanks": channel_shanks,
        "cluster_Amplitude": cluster_Amplitude,
        "cluster_ContamPct": cluster_ContamPct,
        "cluster_group": cluster_group,
        "cluster_KSLabel": cluster_KSLabel,
        "kept_spikes": kept_spikes,
        "pc_features_ind": pc_features_ind,
        "pc_features": pc_features,
        "similar_templates": similar_templates,
        "spike_clusters": spike_clusters,
        "spike_detection_templates": spike_detection_templates,
        "spike_positions": spike_positions,
        "spike_templates": spike_templates,
        "spike_times": spike_times,
        "templates": templates,
        "templates_ind": templates_ind,
        "whitening_mat": whitening_mat,
        "whitening_mat_data": whitening_mat_data,
        "whitening_mat_inv": whitening_mat_inv
    }

    savemat(save_path, data)

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
