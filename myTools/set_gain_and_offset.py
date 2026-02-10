import numpy as np

def set_gain_and_offset(meta_ap, recording):

    ### Set gain and offset to recording ###
    im_ai_range_max = float(meta_ap.get('imAiRangeMax', 0.6))
    im_ai_range_min = float(meta_ap.get('imAiRangeMin', -0.6))
    im_max_int = int(meta_ap.get('imMaxInt', 512))
    ap_gain = float(meta_ap.get('imChan0apGain', 500))

    # μVへの変換係数
    gain_to_uv = (im_ai_range_max - im_ai_range_min) / (2 * im_max_int * ap_gain) * 1e6

    # 現在のrecordingのチャンネル数を取得（SYNCチャンネルを除外した場合は384）
    current_num_channels = recording.get_num_channels()

    # すべてのチャンネルに同じゲインを設定
    gains = np.full(current_num_channels, gain_to_uv, dtype='float32')
    recording.set_channel_gains(gains)

    # オフセットは0に設定
    offsets = np.zeros(current_num_channels, dtype='float32')
    recording.set_channel_offsets(offsets)
    
    return recording