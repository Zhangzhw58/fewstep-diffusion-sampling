# Shared CIFAR-10 diffusion framework

This self-contained folder contains the exact pretrained Diffusers checkpoint and reusable inference helpers for `google/ddpm-cifar10-32` (32x32 unconditional CIFAR-10).

## Contents
- `checkpoints/google-ddpm-cifar10-32/`: model, model config, and original scheduler config at revision `267b167dc01f0e4e61923ea244e8b988f84deb80`.
- `sampler.py`: model loader, DPM-Solver++ / UniPC scheduler factory, batched reproducible sampling, and a power-warped log-SNR timestep builder.

## Use
Requirements: Python 3.10+ (tested with 3.13), PyTorch with a CUDA build for GPU use (tested with 2.9.0+cu128), Diffusers 0.31.0, NumPy, and Pillow. Install PyTorch separately for your machine, then install `requirements.txt`. This package contains the checkpoint and does not need to fetch model files from the internet. Add this folder to `PYTHONPATH` or import `sampler.py` directly. Example:
```python
from sampler import load_pipeline, make_scheduler, sample_batch
pipe = load_pipeline('checkpoints/google-ddpm-cifar10-32', 'cuda')
scheduler = make_scheduler(pipe, 'unipc', solver_order=2)
images = sample_batch(pipe, scheduler, seeds=[1, 2, 3], n_steps=10, device='cuda')
```
To run on CPU, pass `device='cpu'` to both `load_pipeline` and `sample_batch` (sampling will be slower). Returned values are PIL images. `sample_batch` resets the scheduler timestep grid each call; create a fresh scheduler for independent or parallel jobs.
The DPM-Solver++ experiment uses only `make_scheduler(pipe, 'dpmpp')`. The UniPC path is an interface for the teammate's separate work and is not part of my experiment results.

The included `unipc_handoff_smoke.png` was produced by one 5-step test to verify this API. It is not part of the DPM-Solver++ experiment results.
