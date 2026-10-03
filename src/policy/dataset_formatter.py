"""
Dataset Formatter for LeRobot & SmolVLA
Aggregates paired simulator rollouts and structures them into
the Hugging Face LeRobot format for VLA fine-tuning and evaluation.
"""

import h5py
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict


def build_lerobot_hdf5_dataset(
    paired_episodes_dir: str,
    output_hdf5_path: str,
    task_description: str = "pick up the object and place it on the target",
):
    """
    Combines recorded paired episodes into a single HDF5 dataset compatible with LeRobot / VLA.
    """
    episodes_path = Path(paired_episodes_dir)
    npz_files = sorted(list(episodes_path.glob("*.npz")))

    if not npz_files:
        print(f"[-] No paired episode files found in {paired_episodes_dir}")
        print("    Run 'src/simulation/replay_trajectory.py --out_dataset ...' first.")
        return

    out_file = Path(output_hdf5_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Packaging {len(npz_files)} episodes into HDF5 format -> {out_file}")

    with h5py.File(str(out_file), "w") as root:
        data_grp = root.create_group("data")
        total_frames = 0

        for ep_idx, npz_f in enumerate(npz_files):
            ep_data = np.load(npz_f)
            sim_frames = ep_data["sim_frames"]   # Shape: (T, 256, 256, 3), uint8
            actions = ep_data["actions"]         # Shape: (T, 7), float32
            T = len(actions)

            ep_grp = data_grp.create_group(f"demo_{ep_idx}")
            ep_grp.create_dataset("actions", data=actions, dtype="float32")
            ep_grp.create_dataset("obs/agentview_rgb", data=sim_frames, dtype="uint8", compression="gzip")
            
            if "robot_eef_pos" in ep_data and ep_data["robot_eef_pos"] is not None:
                ep_grp.create_dataset("obs/eef_pos", data=ep_data["robot_eef_pos"], dtype="float32")

            ep_grp.attrs["num_samples"] = T
            ep_grp.attrs["language_instruction"] = task_description
            total_frames += T

        data_grp.attrs["total"] = total_frames
        data_grp.attrs["num_episodes"] = len(npz_files)

    print(f"[+] Dataset successfully created with {len(npz_files)} episodes ({total_frames} total frames).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert paired episodes to LeRobot HDF5 dataset")
    parser.add_argument("--episodes_dir", type=str, default="data/paired_episodes", help="Dir with paired episode npz files")
    parser.add_argument("--output_hdf5", type=str, default="data/dataset.hdf5", help="Output HDF5 path")
    parser.add_argument("--task", type=str, default="pick up the mug and place it on the coaster", help="Task description")
    args = parser.parse_args()

    build_lerobot_hdf5_dataset(args.episodes_dir, args.output_hdf5, args.task)
