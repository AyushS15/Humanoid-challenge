"""
Comparative Experiment: 3 Approaches to Hand Landmark & Grasp Tracking

Evaluates:
- Approach 1: Full-Frame Letterboxing (FOV expansion, standard L4-L8 distance)
- Approach 2: Cropped Frame + Multi-Point Index Fallback (L8 -> L7 -> L6 + jump detection)
- Approach 3: Combined Hybrid (Letterboxed FOV + Multi-Point Fallback + Grasp Hysteresis)

Metrics:
1. Detection Rate (% of frames tracked)
2. Edge Clipping Occurrences (frames where landmarks touch border)
3. Abrupt Jump Occurrences (frame-to-frame jump > threshold)
4. Grasp Chatter (erratic state transitions)
5. Representative grasp metrics across reach, grasp, and lift phases
"""

import cv2
import json
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"


def run_experiment_on_video(video_path: str = "media/Vid_0.mp4"):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    # 1. Approach 1: Full-Frame Letterbox + Standard L4-L8
    # 2. Approach 2: Cropped Frame + Multi-Point Fallback
    # 3. Approach 3: Letterboxed Frame + Multi-Point Fallback + Hysteresis

    results = {
        "Approach 1 (Letterbox FOV)": {"detected": 0, "edge_clipping": 0, "jumps": 0, "state_flips": 0, "dists": []},
        "Approach 2 (Cropped + MultiPoint)": {"detected": 0, "edge_clipping": 0, "jumps": 0, "state_flips": 0, "dists": []},
        "Approach 3 (Hybrid: Letterbox + MultiPoint + Hysteresis)": {"detected": 0, "edge_clipping": 0, "jumps": 0, "state_flips": 0, "dists": []},
    }

    # Tracking states
    landmarker_1 = vision.HandLandmarker.create_from_options(options)
    landmarker_2 = vision.HandLandmarker.create_from_options(options)

    # Padding parameters for Letterbox
    max_dim = max(width, height)
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left

    # Crop parameters for Cropped Frame
    crop_size = min(width, height)
    y_crop_offset = int((height - crop_size) * 0.65) if height > width else 0
    x_crop_offset = (width - crop_size) // 2

    last_dist_1 = None
    last_dist_2 = None
    last_dist_3 = None

    last_idx_tip_2 = None
    last_idx_tip_3 = None

    state_1 = -1
    state_2 = -1
    state_3 = -1

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = int(frame_idx * 1000 / fps)

        # ------------------------------------------------------------------
        # Frame A: Letterboxed (1024x1024 -> 256x256)
        # ------------------------------------------------------------------
        padded = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[30, 30, 30])
        rgb_letterbox = cv2.cvtColor(cv2.resize(padded, (256, 256)), cv2.COLOR_BGR2RGB)
        mp_img_lb = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_letterbox)
        res_lb = landmarker_1.detect_for_video(mp_img_lb, ts_ms)

        # ------------------------------------------------------------------
        # Frame B: Cropped (576x576 -> 256x256)
        # ------------------------------------------------------------------
        cropped = frame[y_crop_offset : y_crop_offset + crop_size, x_crop_offset : x_crop_offset + crop_size]
        rgb_cropped = cv2.cvtColor(cv2.resize(cropped, (256, 256)), cv2.COLOR_BGR2RGB)
        mp_img_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cropped)
        res_crop = landmarker_2.detect_for_video(mp_img_crop, ts_ms)

        # === EVALUATE APPROACH 1: Letterbox + Pure Tip (L4-L8) ===
        if res_lb.hand_landmarks:
            results["Approach 1 (Letterbox FOV)"]["detected"] += 1
            lm = res_lb.hand_landmarks[0]
            p4 = np.array([lm[4].x, lm[4].y, lm[4].z])
            p8 = np.array([lm[8].x, lm[8].y, lm[8].z])
            dist1 = float(np.linalg.norm(p4 - p8))
            results["Approach 1 (Letterbox FOV)"]["dists"].append(dist1)

            # Check clipping
            if lm[8].y < 0.03 or lm[8].y > 0.97 or lm[8].x < 0.03 or lm[8].x > 0.97:
                results["Approach 1 (Letterbox FOV)"]["edge_clipping"] += 1

            # Check jump
            if last_dist_1 is not None and abs(dist1 - last_dist_1) > 0.12:
                results["Approach 1 (Letterbox FOV)"]["jumps"] += 1
            last_dist_1 = dist1

            # Threshold state (no hysteresis)
            new_s1 = 1 if dist1 < 0.22 else -1
            if new_s1 != state_1 and state_1 != 0:
                results["Approach 1 (Letterbox FOV)"]["state_flips"] += 1
            state_1 = new_s1

        # === EVALUATE APPROACH 2: Cropped + MultiPoint Fallback ===
        if res_crop.hand_landmarks:
            results["Approach 2 (Cropped + MultiPoint)"]["detected"] += 1
            lm = res_crop.hand_landmarks[0]
            p4 = np.array([lm[4].x, lm[4].y, lm[4].z])
            p8 = np.array([lm[8].x, lm[8].y, lm[8].z])
            p7 = np.array([lm[7].x, lm[7].y, lm[7].z])
            p6 = np.array([lm[6].x, lm[6].y, lm[6].z])

            # Check clipping in cropped frame
            if lm[8].y < 0.03 or lm[8].y > 0.97:
                results["Approach 2 (Cropped + MultiPoint)"]["edge_clipping"] += 1

            # Multi-point fallback
            is_abrupt = False
            if last_idx_tip_2 is not None and np.linalg.norm(p8 - last_idx_tip_2) > 0.12:
                is_abrupt = True
            if lm[8].y < 0.05 or lm[8].y > 0.95:
                is_abrupt = True

            if not is_abrupt:
                last_idx_tip_2 = p8.copy()
                dist2 = float(min(np.linalg.norm(p4 - p8), np.linalg.norm(p4 - p7)))
            else:
                dist2 = float(min(np.linalg.norm(p4 - p7), np.linalg.norm(p4 - p6)))

            results["Approach 2 (Cropped + MultiPoint)"]["dists"].append(dist2)

            if last_dist_2 is not None and abs(dist2 - last_dist_2) > 0.12:
                results["Approach 2 (Cropped + MultiPoint)"]["jumps"] += 1
            last_dist_2 = dist2

            new_s2 = 1 if dist2 < 0.22 else -1
            if new_s2 != state_2 and state_2 != 0:
                results["Approach 2 (Cropped + MultiPoint)"]["state_flips"] += 1
            state_2 = new_s2

        # === EVALUATE APPROACH 3: Hybrid (Letterbox + MultiPoint + Hysteresis) ===
        if res_lb.hand_landmarks:
            results["Approach 3 (Hybrid: Letterbox + MultiPoint + Hysteresis)"]["detected"] += 1
            lm = res_lb.hand_landmarks[0]
            p4 = np.array([lm[4].x, lm[4].y, lm[4].z])
            p8 = np.array([lm[8].x, lm[8].y, lm[8].z])
            p7 = np.array([lm[7].x, lm[7].y, lm[7].z])
            p6 = np.array([lm[6].x, lm[6].y, lm[6].z])

            if lm[8].y < 0.03 or lm[8].y > 0.97:
                results["Approach 3 (Hybrid: Letterbox + MultiPoint + Hysteresis)"]["edge_clipping"] += 1

            is_abrupt = False
            if last_idx_tip_3 is not None and np.linalg.norm(p8 - last_idx_tip_3) > 0.12:
                is_abrupt = True
            if lm[8].y < 0.04 or lm[8].y > 0.96:
                is_abrupt = True

            if not is_abrupt:
                last_idx_tip_3 = p8.copy()
                dist3 = float(min(np.linalg.norm(p4 - p8), np.linalg.norm(p4 - p7)))
            else:
                dist3 = float(min(np.linalg.norm(p4 - p7), np.linalg.norm(p4 - p6)))

            results["Approach 3 (Hybrid: Letterbox + MultiPoint + Hysteresis)"]["dists"].append(dist3)

            if last_dist_3 is not None and abs(dist3 - last_dist_3) > 0.12:
                results["Approach 3 (Hybrid: Letterbox + MultiPoint + Hysteresis)"]["jumps"] += 1
            last_dist_3 = dist3

            # Hysteresis: close at 0.19, open at 0.26
            if state_3 < 0:
                if dist3 < 0.19:
                    state_3 = 1
                    results["Approach 3 (Hybrid: Letterbox + MultiPoint + Hysteresis)"]["state_flips"] += 1
            else:
                if dist3 > 0.26:
                    state_3 = -1
                    results["Approach 3 (Hybrid: Letterbox + MultiPoint + Hysteresis)"]["state_flips"] += 1

    cap.release()

    # Summary statistics
    summary = {}
    for name, data in results.items():
        dists = data["dists"]
        summary[name] = {
            "Detection Frames": f"{data['detected']} / {total_frames} ({data['detected']/total_frames*100:.1f}%)",
            "Border Clipping Frames": data["edge_clipping"],
            "Abrupt Distance Jumps (>0.12)": data["jumps"],
            "State Flips / Chatter": data["state_flips"],
            "Mean Distance": f"{np.mean(dists):.3f}" if dists else "N/A",
            "Min Distance (Grasp)": f"{np.min(dists):.3f}" if dists else "N/A",
            "Max Distance (Reach)": f"{np.max(dists):.3f}" if dists else "N/A",
        }

    return summary


if __name__ == "__main__":
    res = run_experiment_on_video("media/Vid_0.mp4")
    print(json.dumps(res, indent=2))
