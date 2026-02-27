import os
import csv
import matplotlib.pyplot as plt

import dynosam_utils.evaluation.evaluation_lib as eval
from dynosam_utils.evaluation.core.plotting import startup_plotting

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
            timestamps.append(float(row["timestamp"]))
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



def plot_experiments(experiment_folders):
    # =============================
    # Figure 1: Solve + Tracks
    # =============================
    fig1 = plt.figure()
    ax_main = fig1.add_subplot(111)

    ax_main.set_title("Solve Time Per Frame")
    ax_main.set_xlabel("Frame id")
    ax_main.set_ylabel("Time [ms]")

    # =============================
    # Figure 2: Translation Error
    # =============================
    fig2 = plt.figure()
    ax_trans = fig2.add_subplot(111)
    ax_trans.set_title("Average Object Translation Error vs Frame id")
    ax_trans.set_xlabel("Frame id")
    ax_trans.set_ylabel("Avg Translation Error")

    # =============================
    # Figure 3: Rotation Error
    # =============================
    fig3 = plt.figure()
    ax_rot = fig3.add_subplot(111)
    ax_rot.set_title("Average Object Rotation Error vs Frame id")
    ax_rot.set_xlabel("Frame id")
    ax_rot.set_ylabel("Avg Rotation Error")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

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

        motion_eval = dataset_eval.create_motion_error_evaluator(data_files)
        motions_errors = motion_eval.get_mottion_error()

        # === Compute averaged object errors ===
        trans_timestamps, avg_trans_errors = accumulate_average_error(
            motions_errors, "trans"
        )

        rot_timestamps, avg_rot_errors = accumulate_average_error(
            motions_errors, "rot"
        )

        rmse_trans_error = rmse(np.array(avg_trans_errors))
        rmse_rot_error = np.mean(np.array(avg_rot_errors))

        # === Load solve/tracks ===
        timestamps, solve_times, number_tracks = load_csv(csv_path)
        combined = sorted(zip(timestamps, solve_times, number_tracks))
        timestamps, solve_times, number_tracks = zip(*combined)

        label = os.path.basename(os.path.normpath(folder))
        color = color_cycle[i % len(color_cycle)]

        print("--------------------RMSE Error-------------------")
        print(f"Sequence {label}:")
        print(f"RMSE (ME) Translation: {rmse_trans_error} Rotation: {rmse_rot_error}")

        # === Plot solve + tracks ===
        ax_main.plot(
            timestamps,
            solve_times,
            linestyle="-",
            color=color,
            label=f"{label}"
        )

        # ax_main.plot(
        #     timestamps,
        #     number_tracks,
        #     linestyle="--",
        #     color=color,
        #     label=f"{label} (number_tracks)"
        # )

        # === Plot averaged translation error ===
        ax_trans.plot(
            trans_timestamps,
            avg_trans_errors,
            color=color,
            linestyle="-",
            label=label
        )

        # === Plot averaged rotation error ===
        ax_rot.plot(
            rot_timestamps,
            avg_rot_errors,
            color=color,
            linestyle="-",
            label=label
        )

    ax_main.legend()
    ax_trans.legend()
    ax_rot.legend()

    ax_main.grid(True)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    plt.show()

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
        "/root/results/frontend_filtering/kitti00_FS",
        "/root/results/frontend_filtering/kitti00_SS",
        "/root/results/frontend_filtering/kitti00_EIF",
        "/root/results/frontend_filtering/kitti00_PnP",
    ]

    plot_experiments(experiment_folders)
