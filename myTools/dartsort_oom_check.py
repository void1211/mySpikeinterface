"""
GMM refinement の OOM リスクを gmm_max_spikes / core_radius から概算するヘルパー。
DARTsort の core_features と TMM のメモリを粗く見積もり、GPU メモリと比較する。
"""

import numpy as np
import torch


def _n_core_channels(geom: np.ndarray, core_radius: float) -> int:
    """core_radius (μm) から各チャンネルの最大近傍数を返す（make_channel_index と同じ論理）"""
    if core_radius <= 0 or np.isinf(core_radius):
        return geom.shape[0]
    from scipy.spatial.distance import pdist, squareform
    neighbors = squareform(pdist(geom, metric="minkowski", p=2)) <= core_radius
    return int(np.max(np.sum(neighbors, axis=1)))


def estimate_refinement_memory_gb(
    geom: np.ndarray,
    gmm_max_spikes: int,
    core_radius: float,
    tpca_rank: int = 8,
    safety_factor: float = 5.0,
) -> float:
    """
    GMM refinement で必要な GPU メモリを概算（GB）。

    - core_features: n_spikes * n_core_channels * tpca_rank * 4 bytes
    - TMM (Coo_invsqrt 等) は近傍サイズに強く依存するため safety_factor で余裕を見る。
    """
    n_core = _n_core_channels(geom, core_radius)
    bytes_per_spike_core = n_core * tpca_rank * 4  # float32
    core_gb = (gmm_max_spikes * bytes_per_spike_core) / (1024**3)
    return core_gb * safety_factor


def check_gmm_refinement_oom(
    geom: np.ndarray,
    gmm_max_spikes: int,
    core_radius: float,
    tpca_rank: int = 8,
    safety_margin: float = 0.7,
    safety_factor: float = 5.0,
) -> tuple[bool, float, float]:
    """
    ​指定した gmm_max_spikes / core_radius で refinement が OOM になりそうかだけ判定する。

    Args:
        geom: チャンネル位置 (N, 2) or (N, 3), μm.
        gmm_max_spikes: sort_params["gmm_max_spikes"].
        core_radius: sort_params["core_radius"]. 数値(μm) または "extract" のときは
                     featurization 相当の大きい半径として 100μm で概算。
        tpca_rank: temporal_pca_rank (デフォルト 8).
        safety_margin: GPU メモリの何割まで使ってよいか (0.7 = 70%).
        safety_factor: refinement 全体のメモリを core の何倍と見積もるか (5 でかなり余裕).

    Returns:
        (ok, estimated_gb, gpu_gb):
        - ok: True なら OOM しなさそう、False なら OOM リスクあり。
        - estimated_gb: 見積もりメモリ (GB).
        - gpu_gb: 利用可能とみなす GPU メモリ (GB)。CUDA がなければ 0。
    """
    if core_radius == "extract" or (isinstance(core_radius, str) and core_radius.lower() == "extract"):
        core_radius_um = 100.0  # featurization_radius 相当で概算
    else:
        core_radius_um = float(core_radius)

    estimated_gb = estimate_refinement_memory_gb(
        geom, gmm_max_spikes, core_radius_um, tpca_rank=tpca_rank, safety_factor=safety_factor
    )

    gpu_gb = 0.0
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    available_gb = gpu_gb * safety_margin
    ok = (gpu_gb == 0) or (estimated_gb <= available_gb)

    return ok, estimated_gb, gpu_gb
