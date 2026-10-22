"""Shared inference helpers for the CIFAR-10 DPM-Solver++ / UniPC comparison."""
from __future__ import annotations
import inspect
import math
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor


def load_pipeline(model_dir: str | Path, device: str = "cuda"):
    """Load a local Diffusers checkpoint; no network download is required."""
    model_dir = Path(model_dir)
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, local_files_only=True, low_cpu_mem_usage=False
    )
    pipe = pipe.to(device)
    pipe.unet.eval()
    return pipe


def make_scheduler(pipe, name: str = "dpmpp", solver_order: int = 2):
    """Build a scheduler from the checkpoint's original DDPM noise config.

    The framework exposes UniPC for the teammate's comparison; this experiment
    executes only the DPM-Solver++ branch.
    """
    cfg = pipe.scheduler.config
    if name.lower() in {"dpmpp", "dpm_solver++", "dpmsolver++"}:
        return DPMSolverMultistepScheduler.from_config(
            cfg, algorithm_type="dpmsolver++", solver_order=solver_order,
            timestep_spacing="linspace", thresholding=False,
        )
    if name.lower() == "unipc":
        return UniPCMultistepScheduler.from_config(
            cfg, solver_order=solver_order, timestep_spacing="linspace",
            thresholding=False,
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def logsnr_power_timesteps(scheduler, n_steps: int, gamma: float) -> list[int]:
    """Create a monotone custom grid by power-warping a uniform log-SNR grid.

    gamma=1 is uniform in log-SNR. gamma values are selected only on the
    validation seeds; the test seeds remain held out.
    """
    alpha_bar = scheduler.alphas_cumprod.detach().cpu().numpy().astype(np.float64)
    alpha_bar = np.clip(alpha_bar, 1e-12, 1.0 - 1e-12)
    logsnr = np.log(alpha_bar) - np.log1p(-alpha_bar)  # t=0 high, t=T-1 low
    n_train = len(logsnr)
    u = np.arange(n_steps, dtype=np.float64) / float(n_steps)
    u = np.power(u, float(gamma))
    target = logsnr[-1] + u * (logsnr[0] - logsnr[-1])
    # xp must increase: reverse the training-time log-SNR and time arrays.
    t_float = np.interp(target, logsnr[::-1], np.arange(n_train - 1, -1, -1))
    ts = np.rint(t_float).astype(np.int64)
    ts = np.clip(ts, 0, n_train - 1)
    # Rounding must not introduce duplicate or ascending time points.
    for i in range(1, len(ts)):
        if ts[i] >= ts[i - 1]:
            ts[i] = ts[i - 1] - 1
    if ts[-1] < 0 or len(set(ts.tolist())) != n_steps:
        raise ValueError("Could not construct a unique descending timestep grid")
    return ts.tolist()


def set_custom_timesteps(scheduler, timesteps, device: str = "cuda") -> None:
    """Install a custom descending timestep grid on DPM-Solver++ or UniPC.

    UniPC only accepts ``num_inference_steps``: 
    we let it reset all internal state for the same number of steps, 
    then replace ``timesteps``/``sigmas`` with values computed exactly the way 
    UniPC computes them for its own grids.
    """
    ts = [int(t) for t in timesteps]
    if "timesteps" in inspect.signature(scheduler.set_timesteps).parameters:
        scheduler.set_timesteps(timesteps=ts, device=device)
        return
    cfg = scheduler.config
    scheduler.set_timesteps(len(ts), device=device)  # resets model_outputs, step index and solver state
    # same arithmetic (float32 torch, then numpy interp) as UniPC's own set_timesteps
    ac = scheduler.alphas_cumprod
    sigmas = np.array(((1 - ac) / ac) ** 0.5)
    sig = np.interp(np.array(ts, dtype=np.int64), np.arange(0, len(sigmas)), sigmas)
    if cfg.final_sigmas_type == "sigma_min":
        sigma_last = ((1 - ac[0]) / ac[0]) ** 0.5
    elif cfg.final_sigmas_type == "zero":
        sigma_last = 0.0
    else:
        raise ValueError(f"unsupported final_sigmas_type: {cfg.final_sigmas_type}")
    scheduler.sigmas = torch.from_numpy(np.concatenate([sig, [sigma_last]]).astype(np.float32))  # UniPC keeps sigmas on CPU
    scheduler.timesteps = torch.tensor(ts, dtype=torch.int64, device=device)
    scheduler.num_inference_steps = len(ts)
    if getattr(scheduler, "solver_p", None) is not None:
        set_custom_timesteps(scheduler.solver_p, ts, device)


def sample_batch(pipe, scheduler, seeds: Iterable[int], n_steps: int,
                 gamma: Optional[float] = None, device: str = "cuda") -> list[Image.Image]:
    """Sample a batch with identical initial-noise seeds across configurations."""
    seeds = [int(x) for x in seeds]
    if not seeds:
        return []
    scheduler.set_timesteps(n_steps, device=device)
    if gamma is not None:
        ts = logsnr_power_timesteps(scheduler, n_steps, gamma)
        set_custom_timesteps(scheduler, ts, device)
    h = w = int(pipe.unet.config.sample_size)
    c = int(pipe.unet.config.in_channels)
    generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
    x = randn_tensor((len(seeds), c, h, w), generator=generators,
                     device=torch.device(device), dtype=torch.float32)
    x = x * scheduler.init_noise_sigma
    with torch.inference_mode():
        for t in scheduler.timesteps:
            model_in = scheduler.scale_model_input(x, t)
            eps = pipe.unet(model_in, t).sample
            x = scheduler.step(eps, t, x).prev_sample
    x = (x / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    images = [Image.fromarray(np.uint8(np.round(img * 255))) for img in x]
    return images

