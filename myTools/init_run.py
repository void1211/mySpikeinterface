import spikeinterface.full as si
import numpy as np
from probeinterface import Probe


def get_recording(meta_ap, bin_path):
    ### Get recording info ###
    sampling_frequency = float(meta_ap['imSampRate'])
    num_channels = int(meta_ap['nSavedChans'])

    ## Get AP recording ###
    recording = si.read_binary(
        file_paths=bin_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype='int16',
        time_axis=0,  # time x channels
        is_filtered=False,
    )
    sync_recording = recording.select_channels([num_channels-1])
    recording = recording.select_channels(np.arange(num_channels-1))

    return recording, sync_recording

def get_probe(meta_ap):
    ### Create probe manually from meta file ###
    # snsGeomMapから位置情報を取得
    geom_str = meta_ap.get('~snsGeomMap', '')
    if not geom_str:
        raise ValueError("snsGeomMap not found in meta file")

    # snsGeomMapを解析
    # 形式: (NP1110,1,0,73)(0:15.5:0:1)(0:57.5:42:1)...
    geom_list = geom_str.split(')')

    # 最初のエントリからプローブ情報を取得
    probe_info = geom_list[0][1:].split(',')  # (NP1110,1,0,73) -> ['NP1110', '1', '0', '73']
    probe_type = probe_info[0]
    num_shanks = int(probe_info[1])
    shank_pitch = float(probe_info[2])
    shank_width = float(probe_info[3])

    # 位置情報を抽出
    positions = []
    shank_ids = []
    activated = []
    for geom_entry in geom_list[1:-1]:  # 最初と最後を除外
        if geom_entry:
            # (0:15.5:0:1) -> ['0', '15.5', '0', '1']
            parts = geom_entry[1:].split(':')
            shank_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            act = int(parts[3])
            positions.append([x, y])
            shank_ids.append(shank_id)
            activated.append(act)

    positions = np.array(positions)
    shank_ids = np.array(shank_ids)
    activated = np.array(activated)

    # 保存されているチャンネルを取得
    saved_chans_str = meta_ap.get('snsSaveChanSubset', 'all')
    if saved_chans_str == 'all':
        n_saved_chans = int(meta_ap.get('nSavedChans', len(positions)))
        saved_chans = np.arange(n_saved_chans)
    else:
        saved_chans = []
        for e in saved_chans_str.split(','):
            if ':' in e:
                start, stop = e.split(':')
                start, stop = int(start), int(stop) + 1
                saved_chans.extend(np.arange(start, stop))
            else:
                saved_chans.append(int(e))
        saved_chans = np.array(saved_chans)

    # プローブオブジェクトを作成
    probe = Probe(ndim=2, si_units='um', model_name='NP1110', manufacturer='imec')

    # コンタクトを設定
    probe.set_contacts(
        positions=positions,
        shapes='square',
        shank_ids=shank_ids,
        shape_params={'width': 5.0}  # NP1110の電極サイズ
    )

    # デバイスチャンネルインデックスを初期設定（元の順序）
    probe.set_device_channel_indices(np.arange(len(positions)))

    return probe


def get_probe_sorted(recording, probe):
    # 深さ順の順序を取得
    order_f, order_r = si.order_channels_by_depth(recording)

    # 深さ順に並び替えたpositionsを作成
    sorted_positions = probe.contact_positions[order_f]
    sorted_shank_ids = probe.shank_ids[order_f] if probe.shank_ids is not None else None

    # 新しいプローブを作成（深さ順に並び替えた位置情報を使用）
    probe = Probe(ndim=2, si_units='um', model_name='NP1110', manufacturer='imec')
    probe.set_contacts(
        positions=sorted_positions,
        shapes='square',
        shank_ids=sorted_shank_ids,
        contact_ids=np.arange(len(sorted_positions)),
        shape_params={'width': 5.0}
    )

    # device_channel_indicesには、深さ順に並び替えた後の各電極が
    # 元のバイナリファイルの何番目の行にあったかを示すインデックスを設定
    # order_f[i]は、深さ順に並び替えた後のi番目の電極が、元のデータのorder_f[i]番目だったことを示す
    probe.set_device_channel_indices(order_f)

    return probe

def set_probe_info(probe, meta_ap):
    # プローブの輪郭を設定
    # probe_sorted.create_auto_shape()
    contour_polygon = [
        [0, 600],    # 左上
        [70, 600],    # 右上
        [70, 0],    # 右下
        [35, -30],     # 先端（頂点）
        [0,  0],   # 左下
        
    ]

    # numpy配列に変換してセット
    contour_array = np.array(contour_polygon)
    probe.set_planar_contour(contour_array)

    # アノテーションを追加
    probe.annotate(
        serial_number=meta_ap.get('imDatPrb_sn'),
        part_number=meta_ap.get('imDatPrb_pn'),
        port=meta_ap.get('imDatPrb_port'),
        slot=meta_ap.get('imDatPrb_slot'),
    )
    return probe


def get_stimtime(meta_obx, bin_path):

    fs_obx = float(meta_obx['obSampRate'])
    n_chan_obx = int(meta_obx['nSavedChans'])

    obx_rec = si.read_binary(
        bin_path,
        sampling_frequency=fs_obx,
        num_channels=n_chan_obx,
        dtype='int16',
        time_axis=0,
        is_filtered=False,
    )

    sig_stim = np.array(obx_rec.select_channels([n_chan_obx-1]).get_traces())
    sig_stim_flat = sig_stim.flatten()
    sig_binary = (sig_stim_flat > 0).astype(int)

    # エッジ検出（差分を取る）
    diff_not_zeros = np.diff(sig_binary)

    # 立ち上がりエッジを検出（0→1の変化）
    edge_rise = np.where(diff_not_zeros == 1)[0] + 1

    # 立ち下がりエッジを検出（1→0の変化）
    edge_fall = np.where(diff_not_zeros == -1)[0] + 1

    # データの最初のサンプルで既に信号が立っている場合の取り逃しを防ぐ
    if len(sig_binary) > 0 and sig_binary[0] > 0:
        edge_rise = np.concatenate([[0], edge_rise])

    # データの最後のサンプルで信号が立っている場合の取り逃しを防ぐ
    if len(sig_binary) > 0 and sig_binary[-1] > 0:
        edge_fall = np.concatenate([edge_fall, [len(sig_binary) - 1]])

    # ソートして重複を除去（念のため）
    edge_rise = np.unique(edge_rise)
    edge_fall = np.unique(edge_fall)

    return edge_rise, edge_fall, fs_obx