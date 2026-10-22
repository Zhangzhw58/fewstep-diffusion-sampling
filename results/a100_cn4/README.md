# DPM-Solver++ vs UniPC — 22-cn4, NVIDIA A100-PCIE-40GB (2026-10-22)

Same protocol for both solvers (`configs/experiment.json`): solver_order 2; 16 validation seeds; gamma ∈ {0.65, 0.8, 1.0, 1.2, 1.4} chosen by pixel MSE to a shared 100-step DPM-Solver++ reference; 5,000 held-out test seeds (1000–5999); clean-fid against CIFAR-10 train 32×32 statistics. diffusers 0.31.0, PyTorch 2.5.1+cu121, batch 32. Timing excludes FID feature extraction.

| Solver | NFE | Schedule | gamma | Clean-FID ↓ | ms / image |
|---|---:|---|---:|---:|---:|
| DPM-Solver++ | 5 | default linspace | — | 58.64 | 2.49 |
| DPM-Solver++ | 5 | log-SNR power | 1.0 | 56.94 | 2.49 |
| UniPC | 5 | default linspace | — | **54.58** | 2.87 |
| UniPC | 5 | log-SNR power | 0.8 | 60.22 | 2.92 |
| DPM-Solver++ | 10 | default linspace | — | 31.68 | 4.73 |
| DPM-Solver++ | 10 | log-SNR power | 1.0 | 23.89 | 4.78 |
| UniPC | 10 | default linspace | — | 31.07 | 5.67 |
| UniPC | 10 | log-SNR power | 0.8 | **22.86** | 5.80 |
| DPM-Solver++ | 20 | default linspace | — | 24.08 | 9.33 |
| DPM-Solver++ | 20 | log-SNR power | 1.0 | 20.04 | 9.33 |
| UniPC | 20 | default linspace | — | 23.88 | 11.32 |
| UniPC | 20 | log-SNR power | 1.0 | **19.62** | 11.38 |

Notes
- Cross-hardware check: DPM-Solver++ on the A100 reproduces the RTX 3070 laptop run (`results/rtx3070_laptop_dpmpp/`) within ±0.08 FID (e.g. 58.64 vs 58.60, 23.89 vs 23.87).
- UniPC beats DPM-Solver++ at every NFE under the same schedule except NFE 5 with the tuned grid; it costs ~15–20 % more time per image here.
- Negative result: at NFE 5 the validation proxy picked gamma = 0.8 for UniPC (MSE 0.01336 vs 0.01380 for gamma = 1.0, a 3 % margin on 16 seeds), but the held-out FID got worse (60.22 vs 54.58 default). The pixel-MSE proxy is not reliable when candidates are this close; report it as such.
- Single run, 5,000 samples per setting: differences below ~1 FID should be treated as within noise.
