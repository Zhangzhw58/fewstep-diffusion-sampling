"""Correctness checks for custom timestep grids (set_custom_timesteps in sampler.py).

1. UniPC: a custom grid equal to UniPC's own default grid must reproduce the default
   sampler bit for bit (same timesteps, same sigmas, identical images).
2. DPM-Solver++: same check through its native ``timesteps=`` argument.
3. UniPC with log-SNR power grids (gamma) runs, uses exactly the requested timesteps,
   and gives images that differ from the default grid.
Optional: --reference path/to/original_sampler.py also checks that every configuration
the original code could run gives identical images with the modified code.

Run from the repository root:
    python scripts/check_custom_timesteps.py
"""
import argparse, importlib.util, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # sampler.py lives at the repository root
import sampler as S  # noqa: E402

SEEDS = list(range(16))


def arr(images):
    return np.stack([np.asarray(im) for im in images])


def run(mod, pipe, name, nfe, gamma):
    sched = mod.make_scheduler(pipe, name)
    imgs = arr(mod.sample_batch(pipe, sched, SEEDS, nfe, gamma, DEV))
    return imgs, [int(t) for t in sched.timesteps], sched.sigmas.clone()


def report(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "checkpoints/google-ddpm-cifar10-32"))
    ap.add_argument("--reference", default=None, help="original sampler.py for a regression check")
    a = ap.parse_args()
    global DEV
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = S.load_pipeline(a.model, DEV)
    ok = True

    for name in ("unipc", "dpmpp"):
        for nfe in (5, 10):
            ref_imgs, ref_ts, ref_sig = run(S, pipe, name, nfe, None)
            # feed the default grid through the custom-grid code path (the one gamma uses)
            orig_builder = S.logsnr_power_timesteps
            S.logsnr_power_timesteps = lambda scheduler, n, g: list(ref_ts)
            try:
                imgs, ts, sig = run(S, pipe, name, nfe, 1.0)
            finally:
                S.logsnr_power_timesteps = orig_builder
            ok &= report(ts == ref_ts and torch.equal(sig, ref_sig) and np.array_equal(imgs, ref_imgs),
                         f"{name} NFE={nfe}: custom grid = default grid reproduces default sampler exactly "
                         f"(timesteps {ts == ref_ts}, sigmas {torch.equal(sig, ref_sig)}, images {np.array_equal(imgs, ref_imgs)})")

    for nfe in (5, 10):
        base, _, _ = run(S, pipe, "unipc", nfe, None)
        for gamma in (0.5, 1.0, 2.0):
            want = S.logsnr_power_timesteps(S.make_scheduler(pipe, "unipc"), nfe, gamma)
            imgs, ts, sig = run(S, pipe, "unipc", nfe, gamma)
            good = ts == want and len(sig) == nfe + 1 and float(sig[-1]) == 0.0 and np.isfinite(imgs).all() \
                and not np.array_equal(imgs, base)
            ok &= report(good, f"unipc NFE={nfe} gamma={gamma}: timesteps {ts}, differs from default grid "
                               f"(mean |diff| = {np.abs(imgs.astype(float) - base).mean():.2f} / 255)")

    if a.reference:
        spec = importlib.util.spec_from_file_location("sampler_reference", a.reference)
        R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(R)
        for name, gamma in (("dpmpp", None), ("dpmpp", 1.0), ("dpmpp", 2.0), ("unipc", None)):
            for nfe in (5, 10):
                new, _, _ = run(S, pipe, name, nfe, gamma)
                old, _, _ = run(R, pipe, name, nfe, gamma)
                ok &= report(np.array_equal(new, old), f"regression {name} gamma={gamma} NFE={nfe}: identical to original code")

    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
