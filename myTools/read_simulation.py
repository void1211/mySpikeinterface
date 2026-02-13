from pathlib import Path

def get_simulation_path(dir_info):
    root_dir = dir_info["root_dir"]
    # name = dir_info["name"]
    ep = dir_info["ep"]
    cd = dir_info["cd"]

    cars_dir = Path(root_dir) / ("ep"+ep) / ("cd"+cd) / "cars"
    dict_path = {
        "exp": Path(root_dir) / ("ep"+ep) / ("cd"+cd),
        "cars_dir": cars_dir,
        # Recording.load_npz 用（settings/units/contacts 入り）。Recording.save_npz のデフォルトは recording.npz
        "recording": cars_dir / "recording",
        # SpikeInterface 用 ground truth sorting（別形式）
        "ground_truth_sorting_npz": cars_dir / "ground_truth_sorting.npz",
    }
    return dict_path