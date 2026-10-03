# Humanoid Robot Learning Research: Egocentric Hand-to-Panda Sim Policy

This repository provides an end-to-end robot learning system that translates personally collected egocentric phone video into 7D continuous actions for a Franka Emika Panda arm in the **LIBERO** / MuJoCo simulation environment, enabling policy training and evaluation with **SmolVLA** (via Hugging Face LeRobot).

---

## 1. System Overview

```
 [ Real World ]                                    [ Simulation Environment ]
 Smartphone Camera (Egocentric)                    MuJoCo / RoboSuite (Panda Arm)
       │                                                         ▲
       ▼                                                         │ 7D Actions
 MediaPipe 3D Landmark Tracking                            [dx, dy, dz, 0, 0, 0, grip]
       │                                                         │
       ▼                                                         │
 [ΔX_cam, ΔY_cam, ΔZ_cam, Pinch] ──► Kinematic Retargeter ───────┘
```

1. **Egocentric Data Collection**: Smartphone video recorded from a chest-level perspective showing tabletop reach-and-grasp manipulation.
2. **Keypoint & Action Extraction**: Google MediaPipe Hands tracks 21 3D landmarks per frame to calculate 3D wrist displacements and thumb-index pinch distances.
3. **Kinematic Retargeting**: De-rotates the camera pitch angle ($R_x(-\theta)$) and scales camera deltas into the Franka Panda robot's base frame with Exponential Moving Average (EMA) smoothing.
4. **Simulation Replay & Paired Capture**: Executes actions in RoboSuite/LIBERO, producing synchronized side-by-side videos and paired $(Image_{\text{sim}}, Action_{\text{robot}})$ trajectories.
5. **VLA Integration**: Datasets are formatted into LeRobot HDF5 structures for fine-tuning compact Vision-Language-Action models (SmolVLA).

---

## 2. Mathematical Formulation

### A. Coordinate Frame Alignment
Let $v_{\text{cam}} = [\Delta X_{\text{cam}}, \Delta Y_{\text{cam}}, \Delta Z_{\text{cam}}]^T$ be the inter-frame wrist displacement in the camera frame, and let $\theta \approx 45^\circ$ be the downward pitch of the phone camera.

The pitch-corrected vector in the horizontal table plane is given by:
$$v_{\text{level}} = R_x(-\theta) \, v_{\text{cam}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos(-\theta) & -\sin(-\theta) \\ 0 & \sin(-\theta) & \cos(-\theta) \end{bmatrix} \begin{bmatrix} \Delta X_{\text{cam}} \\ \Delta Y_{\text{cam}} \\ \Delta Z_{\text{cam}} \end{bmatrix}$$

Mapping to the Franka Panda robot base frame:
$$\begin{bmatrix} \Delta X_{\text{robot}} \\ \Delta Y_{\text{robot}} \\ \Delta Z_{\text{robot}} \end{bmatrix} = \mathbf{S} \begin{bmatrix} v_{\text{level}}[2] \\ -v_{\text{level}}[0] \\ -v_{\text{level}}[1] \end{bmatrix}$$
Where $\mathbf{S}$ is a scaling gain matrix calibrated to the reach envelope of the Panda arm ($0.8\text{ m}$).

### B. Action Normalization & Smoothing
To remove high-frequency hand tracking noise before sending torques to the Operational Space Controller (OSC):
$$a_t^{\text{pos}} = \alpha \, \Delta p_{\text{robot}, t} + (1 - \alpha) \, a_{t-1}^{\text{pos}}, \quad \alpha = 0.4$$
$$a_t = \text{clip}\left([a_t^{\text{pos}}, 0, 0, 0, g_t], -0.8, 0.8\right) \in \mathbb{R}^7$$
Where $g_t \in \{-1.0, +1.0\}$ represents the binary gripper open/close command derived from the pinch distance metric $\|L_4 - L_8\|_2$.

---

## 3. Repository Structure

```
humanoid-challenge/
├── data/
│   ├── raw_videos/              # Put your phone MP4 recordings here
│   ├── processed_trajectories/  # Extracted landmarks and actions (.npz)
│   └── paired_episodes/         # Sim rollouts paired with human data
├── src/
│   ├── video_processing/
│   │   ├── record_guidelines.md # Recording instructions & framing rules
│   │   └── hand_tracker.py      # MediaPipe 3D landmark extractor
│   ├── retargeting/
│   │   └── hand_to_panda.py     # Kinematic transform to Panda 7D actions
│   ├── simulation/
│   │   ├── libero_runner.py     # Unified RoboSuite / LIBERO environment
│   │   ├── replay_trajectory.py # Sim execution & side-by-side visualizer
│   │   └── test_synthetic_pipeline.py # Self-contained sanity test
│   └── policy/
│       └── dataset_formatter.py # Packages rollouts into LeRobot HDF5
├── notebooks/
│   └── colab_training.ipynb     # Cloud training notebook for SmolVLA
├── media/
│   └── side_by_side_demo.mp4    # Visual demonstration video
├── requirements.txt
└── README.md
```

---

## 4. Quick-Start Guide

### Prerequisites
* Python 3.10+
* macOS (Apple Silicon supported natively) or Ubuntu Linux

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/humanoid-challenge.git
cd humanoid-challenge

# 2. Install dependencies
pip install -r requirements.txt
```

### End-to-End Workflow

#### Step 1: Record Videos
Follow the instructions in [`src/video_processing/record_guidelines.md`](file:///Users/ayush.shrivastava/.gemini/antigravity/scratch/humanoid-challenge/src/video_processing/record_guidelines.md) to record 10–20 short clips on your phone and place them in `data/raw_videos/`.

#### Step 2: Extract 3D Hand Landmarks & Actions
```bash
python src/video_processing/hand_tracker.py --raw_dir data/raw_videos --output_dir data/processed_trajectories
```

#### Step 3: Retarget to Franka Panda 7D Actions
```bash
python src/retargeting/hand_to_panda.py data/processed_trajectories/demo_01_trajectory.npz
```

#### Step 4: Replay in Simulation & Render Comparison Video
```bash
python src/simulation/replay_trajectory.py \
    --actions data/processed_trajectories/demo_01_actions.npz \
    --out_video media/side_by_side_demo_01.mp4 \
    --out_dataset data/paired_episodes/ep_01.npz \
    --task Lift
```

#### Step 5: Format for SmolVLA / LeRobot & Train
Open [`notebooks/colab_training.ipynb`](file:///Users/ayush.shrivastava/.gemini/antigravity/scratch/humanoid-challenge/notebooks/colab_training.ipynb) in Google Colab to run fine-tuning on a free GPU.

---

## 5. Design Decisions: What Worked & What Didn't

### What Worked
1. **Pinch-to-Grasp Thresholding**: Using the Euclidean distance between landmark 4 (thumb tip) and landmark 8 (index tip) proved remarkably robust to hand rotations and lighting changes.
2. **Camera Pitch Compensation**: De-rotating the camera space by $R_x(-45^\circ)$ resolved the natural coupling where reaching forward in egocentric video looks like moving "downward" in pixel coordinates.
3. **EMA Smoothing ($\alpha=0.4$)**: Essential for damping small frame-to-frame keypoint jitter, preventing erratic motor jerks in the Panda arm.

### What Didn't Work & Limitations
1. **Monocular Depth Ambiguity**: Without depth sensors (like LiDAR or stereo cameras), estimating exact metric distance along the optical axis ($Z_{\text{cam}}$) from a phone video exhibits variance when hands move far from the camera.
2. **Direct Visual Transfer**: Feeding human hand frames directly to an end-effector policy caused visual distribution shift (the policy expects metal Panda fingers, not human skin). Generating the paired simulation dataset via retargeting solved this embodiment gap.
