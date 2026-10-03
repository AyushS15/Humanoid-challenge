"""
Hand Landmark Tracker & Pseudo-Action Extractor
Extracts 3D hand keypoints from egocentric phone videos using MediaPipe Hands
and computes relative wrist displacements and pinch gestures.
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class HandTrajectoryExtractor:
    """
    Processes video frames to extract hand 3D landmarks,
    wrist deltas (dx, dy, dz), and normalized pinch metrics.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        pinch_threshold: float = 0.06,
        target_size: Tuple[int, int] = (256, 256),
    ):
        self.pinch_threshold = pinch_threshold
        self.target_size = target_size

        if not MEDIAPIPE_AVAILABLE:
            print("[Warning] MediaPipe is not installed. Run 'pip install mediapipe'.")
            self.mp_hands = None
            self.hands = None
        else:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.mp_draw = mp.solutions.drawing_utils

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        save_annotated_video: bool = True,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extracts trajectory from a single video file.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[*] Processing {video_path.name}: {total_frames} frames @ {fps:.1f} FPS ({width}x{height})")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        annotated_writer = None
        if save_annotated_video:
            annotated_video_path = output_path / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            annotated_writer = cv2.VideoWriter(
                str(annotated_video_path), fourcc, fps, self.target_size
            )

        frames_rgb = []
        wrist_positions = []
        pinch_distances = []
        gripper_states = []
        raw_landmarks_all = []
        timestamps = []

        frame_idx = 0
        last_wrist = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Center-crop to square then resize to target_size
            h, w, _ = frame.shape
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            cropped = frame[top : top + min_dim, left : left + min_dim]
            resized = cv2.resize(cropped, self.target_size)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            t = frame_idx / fps
            timestamps.append(t)

            wrist_pos = np.zeros(3, dtype=np.float32)
            pinch_dist = 0.1
            gripper_cmd = -1.0  # -1 is open, +1 is closed
            landmarks_21 = np.zeros((21, 3), dtype=np.float32)

            if self.hands is not None:
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    for idx, lm in enumerate(hand_landmarks.landmark):
                        landmarks_21[idx] = [lm.x, lm.y, lm.z]

                    # Wrist landmark is index 0
                    wrist_pos = landmarks_21[0].copy()

                    # Thumb tip is 4, Index tip is 8
                    thumb_tip = landmarks_21[4]
                    index_tip = landmarks_21[8]
                    pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))

                    # Discrete or smooth gripper command
                    gripper_cmd = 1.0 if pinch_dist < self.pinch_threshold else -1.0

                    if save_annotated_video and annotated_writer:
                        vis_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        self.mp_draw.draw_landmarks(
                            vis_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                        label = f"GRIPPER: {'CLOSED' if gripper_cmd > 0 else 'OPEN'} (dist: {pinch_dist:.3f})"
                        color = (0, 0, 255) if gripper_cmd > 0 else (0, 255, 0)
                        cv2.putText(vis_bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        annotated_writer.write(vis_bgr)
                else:
                    # Maintain last known position if lost for 1 frame
                    if last_wrist is not None:
                        wrist_pos = last_wrist.copy()
                    if save_annotated_video and annotated_writer:
                        vis_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        annotated_writer.write(vis_bgr)

            frames_rgb.append(rgb_frame)
            wrist_positions.append(wrist_pos)
            pinch_distances.append(pinch_dist)
            gripper_states.append(gripper_cmd)
            raw_landmarks_all.append(landmarks_21)
            last_wrist = wrist_pos

            frame_idx += 1

        cap.release()
        if annotated_writer:
            annotated_writer.release()

        wrist_positions = np.array(wrist_positions, dtype=np.float32)
        gripper_states = np.array(gripper_states, dtype=np.float32)

        # Compute wrist deltas: delta_pos[t] = pos[t] - pos[t-1]
        wrist_deltas = np.zeros_like(wrist_positions)
        if len(wrist_positions) > 1:
            wrist_deltas[1:] = wrist_positions[1:] - wrist_positions[:-1]

        trajectory_data = {
            "timestamps": np.array(timestamps, dtype=np.float32),
            "rgb_frames": np.array(frames_rgb, dtype=np.uint8),
            "wrist_positions": wrist_positions,
            "wrist_deltas": wrist_deltas,
            "pinch_distances": np.array(pinch_distances, dtype=np.float32),
            "gripper_states": gripper_states,
            "landmarks_3d": np.array(raw_landmarks_all, dtype=np.float32),
            "fps": np.float32(fps),
        }

        save_file = output_path / f"{video_path.stem}_trajectory.npz"
        np.savez_compressed(save_file, **trajectory_data)
        print(f"[+] Successfully extracted {frame_idx} steps -> {save_file}")
        return trajectory_data


def process_all_raw_videos(raw_dir: str, output_dir: str):
    """Batch processes all mp4/mov videos in raw_dir."""
    raw_path = Path(raw_dir)
    videos = sorted(list(raw_path.glob("*.mp4")) + list(raw_path.glob("*.mov")))
    if not videos:
        print(f"[-] No .mp4 or .mov files found in {raw_dir}")
        print("    Please follow 'src/video_processing/record_guidelines.md' to record videos.")
        return

    extractor = HandTrajectoryExtractor()
    for video in videos:
        extractor.process_video(str(video), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand landmarks and actions from video")
    parser.add_argument("--video", type=str, default=None, help="Path to single video file")
    parser.add_argument("--raw_dir", type=str, default="data/raw_videos", help="Directory of raw videos")
    parser.add_argument("--output_dir", type=str, default="data/processed_trajectories", help="Output directory")
    args = parser.parse_args()

    if args.video:
        extractor = HandTrajectoryExtractor()
        extractor.process_video(args.video, args.output_dir)
    else:
        process_all_raw_videos(args.raw_dir, args.output_dir)
