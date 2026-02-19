from dynosam_utils.evaluation.tools import load_bson
from dynosam_utils.evaluation.core.plotting import startup_plotting
import matplotlib.pyplot as plt

import sys
import os
import warnings


def flatten_dict(d):
    flat = {}

    def _flatten(current):
        for key, value in current.items():
            if isinstance(value, dict):
                _flatten(value)
            else:
                if key in flat:
                    warnings.warn(
                        f"Key collision detected for '{key}'. "
                        f"Overwriting previous value.",
                        RuntimeWarning,
                    )
                flat[key] = value

    _flatten(d)
    return flat


sequence = "misc"
result_path = f"/root/results/{sequence}/"

logger_prefix = "hybrid_motion_smoother_j"

# find files with logger name
def get_debug_files():
    from pathlib import Path
    output_folder_path = Path(result_path)

    files = []

    for p in output_folder_path.iterdir():
        if p.is_file() and p.name.startswith(logger_prefix):
            files.append(str(p))
    return files

def make_object_debug_map(debug_files):
    """
    Returns:
        dict[str, list[dict]]
        {
            object_name: [ flattened_debug_entry, ... ]
        }
    """
    debug_map = {}

    import re

    def extract_object_id(file_path):
        filename = os.path.basename(file_path)

        match = re.search(r"hybrid_motion_smoother_j(\d+)_debug", filename)
        if not match:
            raise ValueError(f"Could not extract object id from '{filename}'")

        return int(match.group(1))

    for file in debug_files:
        object_id = extract_object_id(file)

        debug_result_list = load_bson(file)[0]['data']

        flattened_list = []
        for debug_result in debug_result_list:
            flattened = flatten_dict(debug_result)
            print(flattened.keys())
            flattened_list.append(flattened)

        flattened_list.sort(key=lambda x: x['timestamp'])

        debug_map[object_id] = flattened_list

    return debug_map

def plot_debug_metric(debug_map, query_key, object_name=None):
    """
    Parameters
    ----------
    debug_map : dict
        Output from make_object_debug_map
    query_key : str
        Field to plot on y-axis
    object_name : str or None
        If None → plot all objects
    """

    if object_name is not None:
        if object_name not in debug_map:
            raise ValueError(f"Object '{object_name}' not found.")

        objects_to_plot = {object_name: debug_map[object_name]}
    else:
        objects_to_plot = debug_map

    fig = plt.figure()
    ax = fig.gca()

    for obj_name, data_list in objects_to_plot.items():

        if not data_list:
            continue

        if query_key not in data_list[0]:
            raise ValueError(
                f"Query key '{query_key}' not found in object '{obj_name}'."
            )

        timestamps = [d["frame_id"] for d in data_list]
        values = [d[query_key] for d in data_list]

        ax.plot(timestamps, values, label=obj_name)

    ax.set_xlabel("frame_id")
    ax.set_ylabel(query_key)
    ax.set_title(f"{query_key} vs frame_id (j={obj_name})")

    if len(objects_to_plot) > 1:
        ax.legend()

    ax.grid(True)
    fig.tight_layout()


debug_map = make_object_debug_map(get_debug_files())

# Plot one object
plot_debug_metric(debug_map, "update_time_ms", object_name=2)
plot_debug_metric(debug_map, "max_clique_size", object_name=2)
# plot_debug_metric(debug_map, "clique", object_name=1)
plot_debug_metric(debug_map, "average_clique_size", object_name=2)
plot_debug_metric(debug_map, "nnz_elements_tree", object_name=2)
plot_debug_metric(debug_map, "average_feature_age", object_name=2)
plot_debug_metric(debug_map, "num_tracks", object_name=2)

# plot_debug_metric(debug_map, "update_time_ms", object_name=1)
# plot_debug_metric(debug_map, "max_clique_size", object_name=1)
plot_debug_metric(debug_map, "num_landmarks_in_smoother", object_name=1)
plot_debug_metric(debug_map, "variables_reeliminated", object_name=1)

# plot_debug_metric(debug_map, "update_time_ms", object_name=2)
# # plot_debug_metric(debug_map, "max_clique_size", object_name=1)
# # # plot_debug_metric(debug_map, "average_clique_size", object_name=2)
# # # plot_debug_metric(debug_map, "nnz_elements_tree", object_name=2)
# # plot_debug_metric(debug_map, "average_feature_age", object_name=1)
# plot_debug_metric(debug_map, "num_tracks", object_name=2)

plt.show()
