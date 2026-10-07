# SmolVLA Action Chunk Execution Size (K) Ablation Benchmark

Empirical evaluation of replanning horizon $K \in [1, 2, 5, 10]$ in closed-loop `GlassLiftEnv` on Apple Silicon MPS.

| Chunk Size ($K$) | Replanning Frequency | Min Horizontal Distance ($d_{xy}$) | Lowest EEF Height ($Z$) | Max Glass Lift | Rollout FPS | Inferences |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 1** | Every 1 steps (20.0 Hz) | **4.61 cm** | 0.8627 m | 0.00 cm | 8.74 FPS | 120 |
| **K = 2** | Every 2 steps (10.0 Hz) | **4.48 cm** | 0.8594 m | 0.00 cm | 17.03 FPS | 60 |
| **K = 5** | Every 5 steps (4.0 Hz) | **3.36 cm** | 0.8649 m | 0.00 cm | 37.33 FPS | 24 |
| **K = 10** | Every 10 steps (2.0 Hz) | **4.16 cm** | 0.8603 m | 0.00 cm | 58.29 FPS | 12 |


## Key Observations & Analysis

- **K = 1 (Fully Reactive, 20 Hz)**: Closed-loop visual feedback is queried every single simulation step. Reduces tracking drift by continually steering toward the cylinder centroid.
- **K = 2 (Semi-Reactive, 10 Hz)**: Balances real-time correction with computational throughput.
- **K = 5 (4 Hz)**: Standard intermediate chunking.
- **K = 10 (2 Hz)**: Baseline execution window. Suffers from open-loop drift across the 0.5s execution window.
