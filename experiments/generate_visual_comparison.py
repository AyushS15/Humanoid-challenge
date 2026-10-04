"""
Generates visual comparison videos and side-by-side keyframes
for Approach 1, Approach 2, and Approach 3 on Vid_0.mp4.
"""

import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"


def render_all_approaches(video_path: str = "media/Vid_0.mp4"):
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

    landmarker_1 = vision.HandLandmarker.create_from_options(options)
    landmarker_2 = vision.HandLandmarker.create_from_options(options)

    # Padding parameters for Letterbox (Approaches 1 & 3)
    max_dim = max(width, height)
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left

    # Crop parameters for Cropped Frame (Approach 2)
    crop_size = min(width, height)
    y_crop_offset = int((height - crop_size) * 0.65) if height > width else 0
    x_crop_offset = (width - crop_size) // 2

    # Writers for individual approach videos
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = Path("media")
    out_dir.mkdir(parents=True, exist_ok=True)

    w1 = cv2.VideoWriter(str(out_dir / "approach_1_letterbox.mp4"), fourcc, fps, (256, 256))
    w2 = cv2.VideoWriter(str(out_dir / "approach_2_cropped.mp4"), fourcc, fps, (256, 256))
    w3 = cv2.VideoWriter(str(out_dir / "approach_3_hybrid.mp4"), fourcc, fps, (256, 256))
    # 3-way combined side-by-side: (768 x 256)
    w_combined = cv2.VideoWriter(str(out_dir / "comparison_3_approaches.mp4"), fourcc, fps, (768, 256))

    state_1 = -1
    state_2 = -1
    state_3 = -1

    last_idx_tip_2 = None
    last_idx_tip_3 = None

    comparison_frames = {}

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = int(frame_idx * 1000 / fps)

        # ------------------------------------------------------------------
        # Frame A: Letterboxed (1024x1024 -> 256x256)
        # ------------------------------------------------------------------
        padded = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[30, 30, 30])
        rgb_lb = cv2.cvtColor(cv2.resize(padded, (256, 256)), cv2.COLOR_BGR2RGB)
        mp_lb = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_lb)
        res_lb = landmarker_1.detect_for_video(mp_lb, ts_ms)

        # ------------------------------------------------------------------
        # Frame B: Cropped (576x576 -> 256x256)
        # ------------------------------------------------------------------
        cropped = frame[y_crop_offset : y_crop_offset + crop_size, x_crop_offset : x_crop_offset + crop_size]
        rgb_crop = cv2.cvtColor(cv2.resize(cropped, (256, 256)), cv2.COLOR_BGR2RGB)
        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        res_crop = landmarker_2.detect_for_video(mp_crop, ts_ms)

        # ==================================================================
        # 1. RENDER APPROACH 1: Letterbox + Pure Tip (L4-L8), Simple Threshold
        # ==================================================================
        vis_1 = cv2.cvtColor(rgb_lb, cv2.COLOR_RGB2BGR)
        dist_1 = 0.4
        if res_lb.hand_landmarks:
            lm = res_lb.hand_landmarks[0]
            for pt in lm:
                cv2.circle(vis_1, (int(pt.x * 256), int(pt.y * 256)), 3, (0, 255, 0), -1)
            p4 = np.array([lm[4].x, lm[4].y, lm[4].z])
            p8 = np.array([lm[8].x, lm[8].y, lm[8].z])
            dist_1 = float(np.linalg.norm(p4 - p8))
            state_1 = 1 if dist_1 < 0.22 else -1
            # Draw pinch line
            cv2.line(vis_1, (int(lm[4].x * 256), int(lm[4].y * 256)), (int(lm[8].x * 256), int(lm[8].y * 256)), (0, 0, 255) if state_1 > 0 else (0, 255, 0), 2)

        lbl_1 = f"Appr 1: {'CLOSED' if state_1 > 0 else 'OPEN'} ({dist_1:.3f})"
        cv2.putText(vis_1, "1. Letterbox FOV Only", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(vis_1, lbl_1, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if state_1 > 0 else (0, 255, 0), 1)
        w1.write(vis_1)

        # ==================================================================
        # 2. RENDER APPROACH 2: Cropped Frame + MultiPoint Fallback
        # ==================================================================
        vis_2 = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
        dist_2 = 0.4
        active_target_2 = None
        used_2 = "L8"
        if res_crop.hand_landmarks:
            lm = res_crop.hand_landmarks[0]
            for pt in lm:
                cv2.circle(vis_2, (int(pt.x * 256), int(pt.y * 256)), 3, (0, 255, 0), -1)
            p4 = np.array([lm[4].x, lm[4].y, lm[4].z])
            p8 = np.array([lm[8].x, lm[8].y, lm[8].z])
            p7 = np.array([lm[7].x, lm[7].y, lm[7].z])
            p6 = np.array([lm[6].x, lm[6].y, lm[6].z])

            is_abrupt_2 = False
            if last_idx_tip_2 is not None and np.linalg.norm(p8 - last_idx_tip_2) > 0.12:
                is_abrupt_2 = True
            if lm[8].y < 0.05 or lm[8].y > 0.95:
                is_abrupt_2 = True

            if not is_abrupt_2:
                last_idx_tip_2 = p8.copy()
                d_tip = float(np.linalg.norm(p4 - p8))
                d_dip = float(np.linalg.norm(p4 - p7))
                if d_tip <= d_dip:
                    dist_2 = d_tip
                    active_target_2 = p8
                    used_2 = "L8"
                else:
                    dist_2 = d_dip
                    active_target_2 = p7
                    used_2 = "L7"
            else:
                dist_2 = float(min(np.linalg.norm(p4 - p7), np.linalg.norm(p4 - p6)))
                active_target_2 = p7
                used_2 = "L7 (fallback)"

            state_2 = 1 if dist_2 < 0.22 else -1
            if active_target_2 is not None:
                cv2.line(vis_2, (int(lm[4].x * 256), int(lm[4].y * 256)), (int(active_target_2[0] * 256), int(active_target_2[1] * 256)), (0, 0, 255) if state_2 > 0 else (0, 255, 0), 2)

        lbl_2 = f"Appr 2: {'CLOSED' if state_2 > 0 else 'OPEN'} ({dist_2:.3f}) [{used_2}]"
        cv2.putText(vis_2, "2. Cropped + MultiPoint", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(vis_2, lbl_2, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if state_2 > 0 else (0, 255, 0), 1)
        w2.write(vis_2)

        # ==================================================================
        # 3. RENDER APPROACH 3: Hybrid (Letterbox + MultiPoint + Hysteresis)
        # ==================================================================
        vis_3 = cv2.cvtColor(rgb_lb, cv2.COLOR_RGB2BGR)
        dist_3 = 0.4
        active_target_3 = None
        used_3 = "L8"
        if res_lb.hand_landmarks:
            lm = res_lb.hand_landmarks[0]
            for pt in lm:
                cv2.circle(vis_3, (int(pt.x * 256), int(pt.y * 256)), 3, (0, 255, 0), -1)
            p4 = np.array([lm[4].x, lm[4].y, lm[4].z])
            p8 = np.array([lm[8].x, lm[8].y, lm[8].z])
            p7 = np.array([lm[7].x, lm[7].y, lm[7].z])
            p6 = np.array([lm[6].x, lm[6].y, lm[6].z])

            is_abrupt_3 = False
            if last_idx_tip_3 is not None and np.linalg.norm(p8 - last_idx_tip_3) > 0.12:
                is_abrupt_3 = True
            if lm[8].y < 0.04 or lm[8].y > 0.96:
                is_abrupt_3 = True

            if not is_abrupt_3:
                last_idx_tip_3 = p8.copy()
                d_tip = float(np.linalg.norm(p4 - p8))
                d_dip = float(np.linalg.norm(p4 - p7))
                if d_tip <= d_dip:
                    dist_3 = d_tip
                    active_target_3 = p8
                    used_3 = "L8"
                else:
                    dist_3 = d_dip
                    active_target_3 = p7
                    used_3 = "L7"
            else:
                dist_3 = float(min(np.linalg.norm(p4 - p7), np.linalg.norm(p4 - p6)))
                active_target_3 = p7
                used_3 = "L7 (fallback)"

            # Hysteresis
            if state_3 < 0:
                if dist_3 < 0.19:
                    state_3 = 1
            else:
                if dist_3 > 0.26:
                    state_3 = -1

            if active_target_3 is not None:
                cv2.line(vis_3, (int(lm[4].x * 256), int(lm[4].y * 256)), (int(active_target_3[0] * 256), int(active_target_3[1] * 256)), (0, 0, 255) if state_3 > 0 else (0, 255, 0), 2)

        lbl_3 = f"Appr 3: {'CLOSED' if state_3 > 0 else 'OPEN'} ({dist_3:.3f}) [{used_3}]"
        cv2.putText(vis_3, "3. Hybrid (LB+Multi+Hyst)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(vis_3, lbl_3, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if state_3 > 0 else (0, 255, 0), 1)
        w3.write(vis_3)

        # ------------------------------------------------------------------
        # Combined 3-way frame [Appr 1 | Appr 2 | Appr 3]
        # ------------------------------------------------------------------
        combined = np.hstack([vis_1, vis_2, vis_3])
        w_combined.write(combined)

        # Save key sample frames at critical timestamps:
        # Frame 85: Reach
        # Frame 105: Grasp
        # Frame 120: Peak Lift (the exact point of clipping bug!)
        if frame_idx in [85, 105, 120]:
            comparison_frames[frame_idx] = combined

    cap.release()
    w1.release()
    w2.release()
    w3.release()
    w_combined.release()

    for idx, img in comparison_frames.items():
        out_img_path = out_dir / f"comparison_frame_{idx}.jpg"
        cv2.imwrite(str(out_img_path), img)
        print(f"[+] Saved comparison frame {idx} -> {out_img_path}")

    print("[+] All videos and comparison frames generated in media/")


if __name__ == "__main__":
    render_all_approaches("media/Vid_0.mp4")
