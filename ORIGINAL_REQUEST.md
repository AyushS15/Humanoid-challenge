# Original User Request

## 2026-10-04T23:10:19Z

Implement and empirically benchmark two computer vision and robot retargeting strategies—(1) CoTracker3 dense point tracking with SVD 6D motion estimation and co-motion grasp correlation, and (2) Hybrid kinematic contact inversion with closed-loop waypoint IK—running on Apple Silicon (MacBook Air M5 GPU/MPS) to retarget 10 real-world glass grasp-and-lift videos into RoboSuite simulation.

Working directory: /Users/ayush.shrivastava/Desktop/ayush_random/humanoid-challenge
Integrity mode: development

## Requirements

### R1. CoTracker3 Dense Point Tracking & Retargeting Pipeline
Implement dense point tracking on the hand and steel glass running locally on Apple Silicon GPU/MPS. The tracker must track dense point grids across video frames, estimate 6D rigid wrist motion over time via SVD/Kabsch alignment, and classify grasp/release transitions using point velocity co-motion correlation (\rho_t) between hand points and object points.

### R2. Hybrid Kinematic Contact & Waypoint IK Pipeline
Implement contact-driven grasping using vertical wrist velocity inflection (dZ/dt \approx 0 on tabletop contact) and closed-loop digital twin waypoint Inverse Kinematics (IK) in RoboSuite. The retargeter must translate real video hand trajectories into collision-free Franka Panda end-effector waypoints (Pre-grasp hover \rightarrow Descend \rightarrow Grasp clamp \rightarrow Vertical lift) and execute closed-loop clamping in simulation.

### R3. Comparative Benchmark & Empirical Evaluation Suite
Implement an automated comparative benchmark that runs both strategies across all 10 recorded demonstration clips (media/Vid_0.mp4 through media/Vid_9.mp4). The suite must quantitatively evaluate grasp timing accuracy, simulation lift success rate (Z_{glass} > 0.85 m off the tabletop), end-effector trajectory smoothness, and runtime latency/FPS on MacBook Air M5.

### R4. Synchronized Multi-View Video Generation
Generate synchronized side-by-side visual comparison videos for all evaluated clips (data/comparisons/Vid_{id}_comparison.mp4) displaying the raw input video with point/landmark tracking overlays alongside the offscreen-rendered RoboSuite simulation replay.

## Verification Resources
- Raw input demonstration videos: media/Vid_0.mp4 through media/Vid_9.mp4
- Existing RoboSuite environment & offscreen renderer: src/simulation/libero_runner.py
- Reference trajectory replay harness: src/simulation/replay_trajectory.py
- Reference algorithm designs: kinematic_actuator_and_cotracker_plan.md and cv_robot_learning_plan.md

## Acceptance Criteria

### Local Execution & Hardware Feasibility
- [ ] Both pipelines run end-to-end locally on MacBook Air M5 utilizing PyTorch MPS / Apple Silicon acceleration without requiring cloud GPUs or external remote servers.
- [ ] RoboSuite offscreen rendering and physics simulation maintain batch throughput (>30 FPS) on macOS arm64.

### Simulation Task Success & Physical Feasibility
- [ ] Both retargeting approaches drive the simulated Panda arm in RoboSuite to grasp and lift the steel glass above the tabletop (Z_{glass} > 0.85 m) sustained for at least 15 simulation steps.
- [ ] Retargeted trajectories respect Panda joint limits, velocity bounds [-1.0, 1.0]^7, and avoid self-collision or table penetration.

### Verification & Empirical Comparison
- [ ] Benchmark script (experiments/compare_cotracker_vs_kinematic.py) executes across all demonstration videos and outputs a structured Markdown/CSV comparison report detailing:
  - Task completion / lift success rate (%)
  - Grasp transition detection latency / error (frames)
  - End-effector trajectory jerk / path length
  - Local inference and retargeting throughput (FPS)
- [ ] Synchronized side-by-side comparison videos are generated in data/comparisons/ for visual inspection of both methods against ground truth.
