from pathlib import Path
import numpy as np
import pprint
import json
import torch
import gc
import traceback
import time
from scipy.io import savemat
import logging
from datetime import datetime
from myTools.conv_ks4_mat import conv_ks4_mat
from kilosort.io import load_ops
from kilosort.run_kilosort import load_sorting
import pandas as pd
from probeinterface.io import write_probeinterface

from myTools.read_spikeglx import get_exp_path, get_geometry, get_channel_map, read_spikeglx_meta
from myTools.set_gain_and_offset import set_gain_and_offset
from myTools.init_run import get_recording, get_probe, get_probe_sorted, set_probe_info, get_stimtime
from probeinterface import Probe
from probeinterface.plotting import plot_probe

import spikeinterface.full as si
# import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
# import spikeinterface.postprocessing as spost
# import spikeinterface.qualitymetrics as sqm
# import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
# import spikeinterface.curation as scur
# import spikeinterface.widgets as sw

from kilosort.plots import plot_drift_amount, plot_drift_scatter, plot_diagnostics, plot_spike_positions

### Select experiment ###
dir_info = {
    "root_dir": r"C:\Users\tanaka-users\NeuronData",
    "name": "ge6w2",
    "ep": "005",
    "run": "002",
    "ng": "0",
    "nt": "0",
}

dict_path = get_exp_path(dir_info)

### Setting sorters ###
do_preprocess = False
do_runsort = True
do_export_phy = False
sorter = "kilosort4"

sort_params = ss.Kilosort4Sorter.default_params()
# Kilosort4用パラメータ
name_thisparam = "default"
sort_params["torch_device"] = "cuda"
sort_params["save_extra_vars"] = True
sort_params["skip_kilosort_preprocessing"] = do_preprocess

### Fetch meta and bin files ###
meta_ap = read_spikeglx_meta(dict_path["ap"]["meta"])
meta_lf = read_spikeglx_meta(dict_path["lf"]["meta"])
meta_obx = read_spikeglx_meta(dict_path["obx"]["meta"])

recording, sync_recording = get_recording(meta_ap, dict_path["ap"]["bin"])
probe = get_probe(meta_ap)
probe = set_probe_info(probe, meta_ap)
recording = recording.set_probe(probe)
probe = get_probe_sorted(recording)
recording = recording.set_probe(probe)
recording = set_gain_and_offset(meta_ap, recording)

### Preprocess recording ###
if do_preprocess:

    print("\n" + "="*20 + " parameter " + "="*20)
    print(f"\n{'='*5} {sorter} {'='*5}")
    pprint.pprint(sort_params)

    params_file = dict_path["exp"] / sorter / name_thisparam / "params.txt"
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(str(params_file), "w", encoding="utf-8") as f:
        json.dump(sort_params, f, indent=4, ensure_ascii=False)

    print("="*5, sorter, "="*5)
    folder = dict_path["exp"] / sorter / name_thisparam
    pp_rec_folder = folder / "pp_rec"

    recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=3000)
    recording_cmr = spre.common_reference(recording_f, reference="global")
    recording_whiten = spre.whiten(recording_cmr, int_scale=200)
    recording_motion = spre.correct_motion(recording_whiten, preset="kilosort_like")
    recording_preprocessed = recording_motion.save(format="binary", folder=pp_rec_folder, overwrite=True)
    recording = recording_preprocessed
else:
    print("skip preprocessing.")

if do_runsort:
    print(f"{'='*5} {sorter} {'='*5}")
    print(f"  {sorter}実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        start_time = time.time()
        
        sorter_output_dir = dict_path["exp"] / sorter / name_thisparam / "sorting"
        sorting = ss.run_sorter(
            sorter_name=sorter,
            folder=sorter_output_dir, 
            remove_existing_folder=True, 
            recording=recording,
            verbose=True,
            **sort_params
            )
            
        elapsed_time = time.time() - start_time
        print(f"  {sorter}実行完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (経過時間: {elapsed_time/60:.1f}分)")

        print(f"  Analyzer作成開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording, format='binary_folder', folder=dict_path["exp"] / sorter / name_thisparam / "analyzer", overwrite=True)
        print(analyzer)
        print("===== Sorting done =====")

    except Exception as e:
        print(f"Error occurred while running {sorter}: {e}")
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  GPUメモリクリア完了 - 割り当て済み: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
else:
    print("skip sorting.")

### export to phy ###
if do_export_phy:
    analyzer = si.load_sorting_analyzer(folder= dict_path["exp"] / sorter / name_thisparam / "analyzer")
    analyzer.compute(['random_spikes', 'waveforms', 'templates'])
    # PC特徴の計算が時間かかってそうなので、Falseにしておく
    sexp.export_to_phy(sorting_analyzer=analyzer, output_folder=dict_path["exp"] / sorter / name_thisparam / "phy", remove_if_exists=True,
                            compute_pc_features=False)
else:
    print("skip export to phy.")


### export to matlab files for ks4 ###
conv_ks4_mat(res_dir=dict_path["exp"] / sorter / name_thisparam / "sorting" / "sorter_output", recording=recording)

### export to probe.json ###
write_probeinterface(dict_path["exp"] / sorter / name_thisparam / "probe.json", probe_or_probegroup=probe)

### plot drift amount, scatter, diagnostics, spike positions ###
ks_dir = dict_path["exp"] / sorter / name_thisparam / "sorting" / "sorter_output"
ops, st, clu, similar_templates, \
    is_ref, est_contam_rate, kept_spikes, \
        tF, Wall, full_st, full_clu, full_amp = \
            load_sorting(ks_dir, device="cuda", load_extra_vars=True)

plot_drift_amount(ops, ks_dir)
plot_drift_scatter(full_st, ks_dir)
plot_diagnostics(Wall, full_clu, ops, ks_dir)
plot_spike_positions(clu, is_ref, ks_dir)


stim_times_rise, stim_times_fall, fs_obx = get_stimtime(meta_obx, dict_path["obx"]["bin"])
savemat(dict_path["exp"] / sorter / name_thisparam / "stim_times.mat", {'stim_times_rise': stim_times_rise, 'stim_times_fall': stim_times_fall, 'fs': fs_obx})
