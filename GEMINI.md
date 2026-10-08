# Humanoid Challenge — Project & Engineering Guidelines

## 1. Workspace Boundaries
- **Active Directory Only**: Operate strictly within the active project repository root (`/Users/ayush.shrivastava/Desktop/ayush_random/humanoid-challenge`).
- **Do Not Touch Secondary Clones**: Never run `git`, read, write, or sync files in secondary clone directories (e.g., `~/Downloads/Humanoid-challenge`) unless the user explicitly requests it for that specific path.

## 2. Repository & Artifact Cleanliness
- **Preservation Whitelist**: Never delete or overwrite:
  - Video recordings in `data/old_videos/` (specifically `Vid_0_ct_comparison copy*.mp4`).
  - Model weights in `outputs/` (`smolvla_glass_expert`, `smolvla_glass_expert_reach_weighted`).
  - Recorded demonstration datasets in `data/lerobot_dataset/` and `data/cotracker_trajectories/`.
- **Pre-Deletion Verification**: Always identify unused files, report them to the user, and obtain explicit consent before removing files.

## 3. Documentation & Technical Communication
- **No Promotional / AI Buzzwords**: Keep all descriptions, documentation, and commit messages concise, empirical, and grounded. Avoid hyperbolic terms like "breakthrough", "high-authority", "scientific introspection", or "flawless".
- **Precise Hardware Scope**: Always clearly identify the robot embodiment as a 7-DoF Franka Emika Panda arm in RoboSuite simulation.
- **Accurate Metric Attribution**:
  - **+7.2 cm**: Demonstration retargeting baseline (digital twin ground truth).
  - **+3.84 cm**: Closed-loop SmolVLA neural policy rollout (dual-axis velocity calibrated).
  - Never conflate demonstration trajectory replay with autonomous neural policy rollout.

## 4. Apple Silicon (MPS) & MuJoCo Offscreen Rendering
- **Unified Memory (UMA) Buffer Protection**: In simulation rollout loops using MuJoCo CGL offscreen rendering on macOS, do NOT invoke `torch.mps.empty_cache()` or `gc.collect()` per step, as Metal GPU unmapping corrupts shared OpenGL offscreen framebuffers.
