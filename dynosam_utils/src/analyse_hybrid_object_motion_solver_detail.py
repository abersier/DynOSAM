import os
import csv
import matplotlib.pyplot as plt

import dynosam_utils.evaluation.evaluation_lib as eval
import dynosam_utils.evaluation.core.plotting as plotting
from dynosam_utils.evaluation.core.plotting import startup_plotting
from dynosam_utils.evaluation.tools import (
    TrajectoryHelper
)

import evo.tools.plot as evo_plot

import sys
import numpy as np

plt.rcdefaults()
startup_plotting()

def load_csv(filepath):
    timestamps = []
    solve_times = []
    number_tracks = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["frame_id"]))
            solve_times.append(float(row["solve_time"]))
            number_tracks.append(float(row["number_tracks"]))

    return timestamps, solve_times, number_tracks

def accumulate_average_error(motions_errors, metric_key):
    """
    motions_errors: dict[object_id] -> metrics dict
    metric_key: "trans" or "rot"

    Returns:
        sorted_timestamps, avg_errors
    """

    from collections import defaultdict

    error_accumulator = defaultdict(list)

    for object_id, metrics in motions_errors.items():
        metric = metrics[metric_key]

        timestamps = metric.timestamps
        errors = metric.error

        for t, e in zip(timestamps, errors):
            error_accumulator[t].append(e)

    # Compute average per timestamp
    avg_errors = []
    sorted_timestamps = sorted(error_accumulator.keys())

    for t in sorted_timestamps:
        values = error_accumulator[t]
        avg_errors.append(sum(values) / len(values))

    return sorted_timestamps, avg_errors


def rmse(error):
    return np.sqrt((error ** 2).mean())



def plot_experiments(experiment_folders, title, select_objects = None):
    # =============================
    # Figure 1: Solve + Tracks
    # =============================
    fig1 = plt.figure()
    ax_main = fig1.add_subplot(111)

    ax_main.set_title("Solve Time Per Frame")
    ax_main.set_xlabel("Frame id [-]")
    ax_main.set_ylabel("Time [ms]")

    # =============================
    # Figure 2: Translation Error
    # =============================
    fig2 = plt.figure()
    ax_trans = fig2.add_subplot(111)
    ax_trans.set_title("Average Motion Error (ME) Translation")
    ax_trans.set_xlabel("Frame id [-]")
    ax_trans.set_ylabel(r"Avg Translation Error (m)")

    # =============================
    # Figure 3: Rotation Error
    # =============================
    fig3 = plt.figure()
    ax_rot = fig3.add_subplot(111)
    ax_rot.set_title(f"Average Motion Error (ME) Rotation")
    ax_rot.set_xlabel("Frame id [-]")
    ax_rot.set_ylabel(r"Avg Rotation Error ($^\circ$)")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    object_pose_map = {}
    objects_plotted = set()
    trajectory_helper = TrajectoryHelper()

    for i, folder in enumerate(experiment_folders):
        csv_path = os.path.join(folder, "hybrid_motion_solver_details.csv")

        dataset_eval = eval.DatasetEvaluator(folder)
        data_files = dataset_eval.make_data_files("frontend")

        if not data_files.check_is_dynosam_results():
            print(f"Invalid data file {data_files}")
            sys.exit(0)

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue

        label = os.path.basename(os.path.normpath(folder))


        motion_eval = dataset_eval.create_motion_error_evaluator(data_files)
        # motions_errors = motion_eval.get_mottion_error()

        object_poses = motion_eval.object_poses_traj

        for object_id, trajectory  in object_poses.items():
            if select_objects is None or (select_objects is not None and object_id in select_objects):
                print(trajectory)
                trajectory_helper.append(trajectory)
                object_pose_map[label + "_obj" + str(object_id)] = trajectory
                objects_plotted.add(object_id)

        # # # === Compute averaged object errors ===
        # trans_timestamps, avg_trans_errors = accumulate_average_error(
        #     motions_errors, "trans"
        # )

        # rot_timestamps, avg_rot_errors = accumulate_average_error(
        #     motions_errors, "rot"
        # )

        # rmse_trans_error = rmse(np.array(avg_trans_errors))
        # rmse_rot_error = np.mean(np.array(avg_rot_errors))

        # === Load solve/tracks ===
        timestamps, solve_times, number_tracks = load_csv(csv_path)
        combined = sorted(zip(timestamps, solve_times, number_tracks))
        timestamps, solve_times, number_tracks = zip(*combined)

        color = color_cycle[i % len(color_cycle)]

        # print("--------------------RMSE Error-------------------")
        # print(f"Sequence {label}:")
        # print(f"RMSE (ME) Translation: {rmse_trans_error} Rotation: {rmse_rot_error}")

        # === Plot solve + tracks ===
        ax_main.plot(
            timestamps,
            solve_times,
            linestyle="-",
            color=color,
            label=f"{label}"
        )

        ax_main.set_yscale("log")

        # ax_main.plot(
        #     timestamps,
        #     number_tracks,
        #     linestyle="--",
        #     color=color,
        #     label=f"{label} (number_tracks)"
        # )

        # === Plot averaged translation error ===
        # ax_trans.plot(
        #     trans_timestamps,
        #     avg_trans_errors,
        #     color=color,
        #     linestyle="-",
        #     label=label
        # )

        # # === Plot averaged rotation error ===
        # ax_rot.plot(
        #     rot_timestamps,
        #     avg_rot_errors,
        #     color=color,
        #     linestyle="-",
        #     label=label
        # )

    real_time_value_ms = 50
    ax_main.axhline(real_time_value_ms, color='red', linestyle='--')

    # Annotation
    # Small vertical offset (in data units)
    offset = 0.02 * (ax_main.get_ylim()[1] - ax_main.get_ylim()[0])

    # Text just above line
    ax_main.text(
        0.98, real_time_value_ms + 0.1,
        f'Realtime threshold',
        color='red',
        ha='right',
        va='bottom',
        transform=ax_main.get_yaxis_transform()
    )

    ax_main.legend()
    ax_trans.legend()
    ax_rot.legend()

    ax_main.grid(True)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    trajectory_fig = plt.figure()
    # trajectory_fig.add_subplot(111, projection="3d")
    traj_fix = plotting.plot_object_trajectories(
        trajectory_fig,
        object_pose_map,
        None,
        evo_plot.PlotMode.xyz
    )
    # traj_fix.get_legend().remove()
    # trajectory_helper.set_ax_limits(traj_fix, evo_plot.PlotMode.xyz)


    traj_fix.set_xlabel("x [m]")
    traj_fix.set_ylabel("y [m]")
    # traj_fix.set_zlabel("z [m]")
    traj_fix.set_title(f"Trajectory comparison (object = {objects_plotted})")
    trajectory_fig.tight_layout()


    file_path = title + "_doo_comparusons"

    fig1.savefig(file_path + "_solve_time.jpg")
    # trajectory_fig.savefig(file_path + "_trajectories.jpg")
    # fig2.savefig(file_path + "_me_t.jpg")
    # fig3.savefig(file_path + "_me_r.jpg")

    # plt.show()

if __name__ == "__main__":
    experiment_folders = [
        # "/root/results/frontend_filtering/kitti04_FS",
        # "/root/results/frontend_filtering/omd_SS",
        # "/root/results/frontend_filtering/omd_FS",
        # "/root/results/frontend_filtering/omd_EIF",
        # "/root/results/frontend_filtering/omd_PnP",
        # "/root/results/frontend_filtering/kitti20_FS",
        # "/root/results/frontend_filtering/kitti20_EIF",
        # "/root/results/frontend_filtering/kitti20_PnP",
        # "/root/results/frontend_filtering/kitti04_FS",
        # "/root/results/frontend_filtering/kitti04_SS",
        # "/root/results/frontend_filtering/kitti04_EIF",
        # "/root/results/frontend_filtering/kitti04_PnP",
        "/root/results/frontend_filtering/tech_lab_1_FS",
        "/root/results/frontend_filtering/tech_lab_1_SS",
        "/root/results/frontend_filtering/tech_lab_1_EIF",
        "/root/results/frontend_filtering/tech_lab_1_PnP",
    ]

    plot_experiments(experiment_folders, "/root/results/frontend_filtering/tech_lab_1", select_objects=[2])
