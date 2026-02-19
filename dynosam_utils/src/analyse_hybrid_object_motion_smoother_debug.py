from dynosam_utils.evaluation.tools import load_bson
from dynosam_utils.evaluation.core.plotting import startup_plotting
import matplotlib.pyplot as plt

import sys


sequence = "test_kitti"
result_path = f"/root/results/frontend_filtering/{sequence}/"

logger_prefix = "hybrid_motion_smoother_j"

# find files with logger name
def get_debug_files():
    from pathlib import Path
    output_folder_path = Path(result_path)



    for p in output_folder_path.iterdir():
        if p.is_file() and p.name.startswith(logger_prefix):


# results = load_bson("/root/results/DynoSAM/incremental_kitti_00_test/parallel_isam2_results.bson")[0]['data']
# results = load_bson("/root/results/DynoSAM/incremental_test/parallel_isam2_results.bson")[0]['data']

# per object at frame how many variables were involved

object_map = {}

plt.rcdefaults()
startup_plotting(40, line_width=3.0)



for object_id, per_frame_results in results.items():
    print(f"Object id {object_id}")

    if object_id == "0":
        continue
    for frame, object_isam_result in per_frame_results.items():
        was_smoother_ok = bool(object_isam_result["was_smoother_ok"])
        if not was_smoother_ok:
            continue
        frame_id = int(object_isam_result["frame_id"])
        full_isam2_result = object_isam_result["isam_result"]
        timing = float(object_isam_result["timing"])
        # collect_involved_variables(full_isam2_result, object_map, frame_id, object_id)

        if object_id not in object_map:
            object_map[object_id] = {
                "frames": [],
                "variables_reeliminated": [],
                "variables_relinearized": [],
                "timing": [],
                "average_clique_size": [],
                "max_clique_size": [],
                "num_variables": [] ,
                "num_landmarks_marked": [] ,
                "num_motions_marked": []   }

        object_map[object_id]["frames"].append(frame_id)
        object_map[object_id]["variables_reeliminated"].append(int(full_isam2_result["variables_reeliminated"]))
        object_map[object_id]["variables_relinearized"].append(int(full_isam2_result["variables_relinearized"]))
        object_map[object_id]["timing"].append(timing)
        object_map[object_id]["average_clique_size"].append(float(object_isam_result["average_clique_size"]))
        object_map[object_id]["max_clique_size"].append(float(object_isam_result["max_clique_size"]))
        object_map[object_id]["num_variables"].append(int(object_isam_result["num_variables"]))
        object_map[object_id]["num_landmarks_marked"].append(int(object_isam_result["num_landmarks_marked"]))
        object_map[object_id]["num_motions_marked"].append(int(object_isam_result["num_motions_marked"]))



        # motion_variable_status = object_isam_result["motion_variable_status"]
        # print(f"frame id {frame_id} {motion_variable_status}")



# Function to plot data
import os
import numpy as np
def plot_variable(variable_name, ylabel, title, ax = None, **kwargs):

    # plot_file_name = file_name + "_" + variable_name + ".pdf"
    # output_file_name = os.path.join(out_data_path, plot_file_name)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    for object_id, data in object_map.items():
        frames = data["frames"]
        values = data[variable_name]

        values_np = np.array(values)
        print(f"Mean value {np.mean(values_np)} for var={variable_name}")
 # plot_file_name = file_name + "_" + variable_name + ".pdf"
    # output_file_name = os.path.join(out_data_path, plot_file_name)
        # Sort frames and associated values
        sorted_indices = sorted(range(len(frames)), key=lambda i: frames[i])
        frames = [frames[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        ax.plot(frames, values, linestyle='-', label=f'Object {object_id}')

    scale = kwargs.get("scale", "linear")
    ax.set_yscale(scale)

    # ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    # ax.legend()
    # fig.savefig(output_file_name)

# per frame average
def compute_average(variable_name):
    values_per_frame = {}
    for object_id, data in object_map.items():
        values = data[variable_name]
        frames = data["frames"]

        sorted_indices = sorted(range(len(frames)), key=lambda i: frames[i])
        frames = [frames[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        for frame, value in zip(frames, values):
            if frame not in values_per_frame:
                values_per_frame[frame] = []
            values_per_frame[frame].append(value)

    # print(values_per_frame)
    means = []
    for _, values in values_per_frame.items():
        means.append(np.mean(np.array(values)))

    print(f"Mean value {variable_name} {np.mean(np.array(means))}, std= {np.std(np.array(means))} for var={variable_name}")

# fig, axes = plt.subplots(2, 2, sharex='col', figsize=(20, 13),  constrained_layout=True)

# ax1 = axes[0, 0]
# ax2 = axes[0, 1]
# ax3 = axes[1, 0]
# ax4 = axes[1, 1]

# ax3.set_xlabel("Frame")
# ax4.set_xlabel("Frame")


# Plot each variable
# # plot_variable("variables_reeliminated", "Variables Reeliminated", "Number of Variables Reeliminated Per Frame Per Object")
# plot_variable("timing", "Timing [ms]", "iSAM2 Update Time", scale="log",ax=ax1)
# plot_variable("average_clique_size", "Number of variables", "Avg. Clique Size", ax=ax2)
# # plot_variable("variables_relinearized", "Variables Relinearized", "Number of Variables Relinearized Per Frame Per Object", ax=ax1)
# # plot_variable("max_clique_size", "Max Clique Size", "Max Clique Size Per Frame Per Object")
# plot_variable("num_landmarks_marked", "Number of variables", r"Landmarks Involved In Update",ax=ax3)
# plot_variable("num_variables", "Number of variables", r"Dynamic Map Size", ax=ax4)

ax1.legend()
# fig.tight_layout()

# fig.savefig(f"/root/results/Dynosam_ecmr2024/{sequence}_parallel_hybrid_bt_analysis.pdf")

# plot_variable("num_motions_marked", "Number Motion Variables ", r"Num Motion Involved In Update")

plt.show()
# compute_average("timing")
