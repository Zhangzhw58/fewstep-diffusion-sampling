"""Smoke test for the group's sampler.py: DPM-Solver++ / UniPC x NFE, sample grids + FID.

Run from the repository root:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py --nfe 5 10 --num 5000
"""
import argparse, json, os, pickle, sys, time, traceback
os.environ.setdefault("OMP_NUM_THREADS", "8")  # shared server: keep CPU use modest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # sampler.py lives at the repository root
from sampler import load_pipeline, make_scheduler, sample_batch  # noqa: E402

# FID assets (NVIDIA EDM code + Inception detector + CIFAR-10 reference statistics), see README.
DEF_EDM = ROOT / "fid/edm"
DEF_DETECTOR = ROOT / "fid/inception-2015-12-05.pkl"
DEF_REF = ROOT / "fid/cifar10-32x32.npz"


def fid_from_feats(feats, ref, dev="cuda"):
    """FID on the GPU: Tr sqrt(S1 S2) = sum sqrt(eig(A S2 A)), A = S1^(1/2) (symmetric, so eigh is exact)."""
    f = torch.from_numpy(feats).to(dev, torch.float64)
    mu = f.mean(0)
    s1 = torch.cov(f.T)
    mu2 = torch.from_numpy(ref["mu"]).to(dev, torch.float64)
    s2 = torch.from_numpy(ref["sigma"]).to(dev, torch.float64)
    w, v = torch.linalg.eigh(s1)
    a = (v * w.clamp(min=0).sqrt()) @ v.T
    tr_sqrt = torch.linalg.eigvalsh(a @ s2 @ a).clamp(min=0).sqrt().sum()
    return float((mu - mu2).square().sum() + torch.trace(s1) + torch.trace(s2) - 2 * tr_sqrt)


def save_grid(images, path, n=8):
    w, h = images[0].size
    grid = Image.new("RGB", (w * n, h * n))
    for i, im in enumerate(images[: n * n]):
        grid.paste(im, ((i % n) * w, (i // n) * h))
    grid.resize((w * n * 2, h * n * 2), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "checkpoints/google-ddpm-cifar10-32"))
    ap.add_argument("--nfe", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--num", type=int, default=5000, help="images per config for FID (0 = grids only)")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--configs", nargs="+", default=["dpmpp", "unipc", "dpmpp:1.0", "unipc:1.0"],
                    help="solver or solver:gamma (gamma = log-SNR power grid)")
    ap.add_argument("--out", default=str(ROOT / "outputs/metrics/smoke.jsonl"))
    ap.add_argument("--edm-root", default=str(DEF_EDM), help="clone of github.com/NVlabs/edm (for dnnlib)")
    ap.add_argument("--detector", default=str(DEF_DETECTOR), help="inception-2015-12-05.pkl")
    ap.add_argument("--ref", default=str(DEF_REF), help="cifar10-32x32.npz reference statistics")
    a = ap.parse_args()
    (ROOT / "outputs/samples").mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    dev = "cuda"
    pipe = load_pipeline(a.model, dev)
    det = None
    if a.num > 0:
        sys.path.insert(0, a.edm_root)  # the detector pickle needs EDM's dnnlib/torch_utils
        with open(a.detector, "rb") as f:
            det = pickle.load(f).to(dev)
        ref = np.load(a.ref)

    for cfg in a.configs:
        name, gamma = (cfg.split(":") + [None])[:2]
        gamma = float(gamma) if gamma else None
        for nfe in a.nfe:
            tag = f"{name}_{'linspace' if gamma is None else f'gamma{gamma:g}'}_nfe{nfe}"
            rec = dict(tag=tag, solver=name, gamma=gamma, nfe=nfe, num=a.num)
            try:
                sched = make_scheduler(pipe, name)
                grid_imgs = sample_batch(pipe, sched, range(64), nfe, gamma, dev)
                save_grid(grid_imgs, ROOT / f"outputs/samples/{tag}.png")
                rec["timesteps"] = [int(t) for t in sched.timesteps]
                if a.num > 0:
                    feats, t0 = [], time.time()
                    for s in range(0, a.num, a.batch):
                        imgs = sample_batch(pipe, sched, range(s, min(s + a.batch, a.num)), nfe, gamma, dev)
                        x = torch.from_numpy(np.stack([np.asarray(im) for im in imgs])).permute(0, 3, 1, 2).to(dev)
                        with torch.no_grad():
                            feats.append(det(x, return_features=True).cpu().numpy().astype(np.float64))
                    rec["sec_per_img"] = round((time.time() - t0) / a.num, 5)
                    rec["fid"] = round(fid_from_feats(np.concatenate(feats), ref), 2)
                rec["status"] = "ok"
            except Exception as e:  # keep going so one broken branch does not hide the others
                rec["status"] = f"ERROR {type(e).__name__}: {e}"
                traceback.print_exc()
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
