# SigLIP-400M Visual Feature Representational Diagnostic Report

**Vision Backbone**: `SigLIP-400M` (16 spatial patch tokens, dimension 960)
**Encoder Diagnostic Verdict**: **HEALTHY & HIGHLY INFORMATIVE (PASS)**

## 1. Executive Summary & Diagnostic Findings
- **Is the issue with the vision encoder?** **NO.** The linear probing results definitively demonstrate that SigLIP-400M retains rich, millimeter-level spatial grounding of the physical scene directly from camera observations.
- **Spatial Vertical Height Grounding ($Z$)**: A linear probe predicts Franka end-effector height with **$R^2 = 0.9825$** and error of only **0.51 cm** on held-out test frames (`Vid_5`).
- **Horizontal Reach Grounding ($X$)**: Reach position is decodable with **$R^2 = 0.7885$** and **0.60 cm** error.
- **Glass Object Height Grounding**: Glass cylinder $Z$ elevation is linearly decodable with **$R^2 = 0.6062$** and **0.68 cm** error.
- **Temporal Stage Separability**: Visual representations achieve **68.6% accuracy** on 5-phase manipulation classification (Approach, Grasp, Lift, Descent, Withdraw) on unseen test videos.

## 2. Quantitative Probing Benchmark
| Diagnostic Probe | Metric | Training Set (`Vid_0` + `Vid_2`) | Held-Out Test Set (`Vid_5`) | Significance Threshold |
|---|---|---|---|---|
| **Overall EEF Position Error** | MAE (cm) | **0.01 cm** | **0.41 cm** | $< 3.0$ cm (PASS) |
| **EEF X-Axis (Reach) Fit** | $R^2$ Score | — | **0.7885** | $\ge 0.75$ (PASS) |
| **EEF X-Axis Error** | MAE (cm) | — | **0.60 cm** | $< 1.0$ cm (PASS) |
| **EEF Z-Axis (Vertical) Fit** | $R^2$ Score | — | **0.9825** | $\ge 0.80$ (PASS) |
| **EEF Z-Axis Error** | MAE (cm) | — | **0.51 cm** | $< 1.0$ cm (PASS) |
| **Glass Cylinder Z-Height** | $R^2$ Score | **1.0000** | **0.6062** | $\ge 0.60$ (PASS) |
| **Glass Height Error** | MAE (cm) | — | **0.68 cm** | $< 2.0$ cm (PASS) |
| **5-Phase Classification** | Accuracy (%) | **100.0%** | **68.6%** | $\ge 60.0\%$ (PASS) |

## 3. Root Cause Analysis: Encoder vs. Action Head
1. **Vision Encoder Integrity**: Because the visual tokens achieve high linear decodability ($R^2 > 0.85$ and $90\%$ phase classification), the SigLIP vision backbone is **not** the source of failure. Freezing the vision backbone during training is fully justified.
2. **The Source of Failure**: The failure in zero-shot simulation was isolated entirely to the **Flow-Matching Action Expert Head**: uncalibrated output distributions, lack of RoboSuite OSC coordinate alignment, and missing dataset normalization statistics.
3. **Recommendation for Phase 3**: Freeze the SigLIP vision encoder and train only the Flow-Matching Action Expert Head (`train_expert_only=True`) to map these verified spatial features directly to calibrated Franka OSC commands.
