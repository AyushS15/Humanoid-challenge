"""
Synthetic End-to-End Pipeline Smoke Test
Generates a synthetic trajectory to verify the retargeting and simulation pipeline
without needing physical phone recordings immediately.
"""

import numpy as np
from pathlib import Path

from src.retargeting.hand_to_panda import HandToPandaRetargeter


def generate_synthetic_trajectory(output_path: str, num_steps: int = 60) -> str:
    """
    Creates a synthetic reach-and-lift trajectory:
    1. Steps 0-25: Move forward (+z_cam) and down (+y_cam) toward object
    2. Steps 25-35: Pinch closed (gripper -> +1.0)
    3. Steps 35-60: Lift up (-y_cam) and retract
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.linspace(0, 3.0, num_steps, dtype=np.float32)
    wrist_deltas = np.zeros((num_steps, 3), dtype=np.float32)
    gripper_states = np.full(num_steps, -1.0, dtype=np.float32)

    # Reach phase
    wrist_deltas[5:25, 1] = 0.03   # downward in cam frame (+Y)
    wrist_deltas[5:25, 2] = 0.04   # forward in cam frame (+Z)

    # Grasp phase
    gripper_states[25:] = 1.0       # close gripper

    # Lift phase
    wrist_deltas[30:50, 1] = -0.04  # upward in cam frame (-Y)
    wrist_deltas[30:50, 2] = -0.01  # slight retract

    # Synthetic RGB placeholder frames
    rgb_frames = np.zeros((num_steps, 256, 256, 3), dtype=np.uint8)

    data = {
        "timestamps": timestamps,
        "wrist_deltas": wrist_deltas,
        "gripper_states": gripper_states,
        "rgb_frames": rgb_frames,
    }
    np.savez_compressed(out_file, **data)
    print(f"[+] Generated synthetic trajectory with {num_steps} steps -> {out_file}")
    return str(out_file)


def test_pipeline():
    print("[*] Running synthetic pipeline test...")
    traj_path = "data/processed_trajectories/synthetic_demo_trajectory.npz"
    generate_synthetic_trajectory(traj_path)

    # Run retargeting
    retargeter = HandToPandaRetargeter(camera_tilt_deg=45.0, position_gain=6.0)
    actions_dict = retargeter.retarget_trajectory(traj_path)

    actions = actions_dict["actions"]
    assert actions.shape == (60, 7), f"Expected shape (60, 7), got {actions.shape}"
    assert np.all(actions[:, 0:6] >= -1.0) and np.all(actions[:, 0:6] <= 1.0), "Actions exceed bounds!"
    print("[+] Retargeting test passed! Action array shape:", actions.shape)
    print("    Sample action step 10 (approach):", np.round(actions[10], 3))
    print("    Sample action step 35 (lift):    ", np.round(actions[35], 3))


if __name__ == "__main__":
    test_pipeline()
