from pathlib import Path

def get_simulation_path(dir_info):
    root_dir = dir_info["root_dir"]
    # name = dir_info["name"]
    ep = dir_info["ep"]
    condition = dir_info["condition"]

    dict_path = {
        "exp": Path(root_dir) / name / exp_name / exp_name,
    }
    return dict_path