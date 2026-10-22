# CIFAR-10 few-step sampling: DPM-Solver++ vs UniPC (group project)

- `shared_framework/` — shared sampler (`sampler.py`: DPM-Solver++ / UniPC, custom log-SNR grids via `set_custom_timesteps`). Download the model with `python scripts/download_model.py`.
- `scripts/run_experiment.py` — DPM-Solver++ experiment (team lead); `scripts/run_unipc.py` — identical protocol for UniPC (zzw). Both read `configs/experiment.json`.
- `scripts/check_custom_timesteps.py` — correctness checks for the custom timestep grids (18 checks, all pass on diffusers 0.31.0).
- `results/rtx3070_laptop_dpmpp/` — the team lead's DPM-Solver++ run (RTX 3070 laptop).
- `results/a100_cn4/` — both solvers on the same A100; see its README for the comparison table.
- Large files (checkpoints, generated images, `.venv`, zips) are not committed; see `.gitignore`.

---

# DPM-Solver CIFAR-10 experiment

This folder contains an isolated experiment for route B: pretrained CIFAR-10 diffusion sampling with DPM-Solver++ only. It does not run UniPC. A reusable scheduler interface and the local checkpoint are under `shared_framework/` for the teammate who will add/run UniPC.

## Hardware and environment
- Tested device: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB VRAM.
- Python environment: `.venv`; it uses the machine's existing PyTorch 2.9.0+cu128 via system site packages, while project-specific packages are kept in `.venv`.
- Diffusers 0.31.0; model `google/ddpm-cifar10-32` at the commit recorded in `configs/experiment.json`.
- Checkpoint is cached locally under `shared_framework/checkpoints/google-ddpm-cifar10-32/`.

## Run order
```powershell
$py = '.\.venv\Scripts\python.exe'
& $py .\scripts\run_experiment.py --phase smoke
& $py .\scripts\run_experiment.py --phase validate
& $py .\scripts\run_experiment.py --phase generate
& $py .\scripts\run_experiment.py --phase fid
& $py .\scripts\make_deliverables.py
```

The completed run uses a 100-step DPM-Solver++ reference, selects a log-SNR power-warp exponent on 16 validation seeds, then generates 5,000 held-out samples for each of six DPM-Solver++ configurations (5/10/20 steps, default and selected schedule). FID is computed against clean-fid's CIFAR-10 train 32×32 reference statistics. The validation score is pixel MSE to the 100-step DPM-Solver++ reference; this tuning proxy is not the final quality metric. `outputs/metrics/experiment_summary.md` documents results and limitations.

The complete experiment can also be started with `--phase all`; phases can be run separately to resume or inspect intermediate artifacts. Run from this directory. Model files are local and the default script does not download them.

## Results and artifacts
- Metrics: `outputs/metrics/fid.csv`, `timing.csv`, `validation_search.csv`, `selected_gamma.json`.
- Charts and report-ready assets: `outputs/metrics/dpmpp_results.png`, `outputs/samples/contact_sheet_48.png`.
- Saved generated images: six subfolders under `outputs/samples/`, 5,000 images per setting.
- Shareable model/sampler package: `shared_framework_for_UniPC.zip` and the source folder `shared_framework/`.

## Reusable interface for UniPC
See `shared_framework/README.md`. It exposes `load_pipeline`, `make_scheduler(pipe, name='unipc')`, `sample_batch`, and custom log-SNR timesteps. The present experiment evaluates DPM-Solver++ only; UniPC was checked with one smoke sample solely to validate the handoff interface.
