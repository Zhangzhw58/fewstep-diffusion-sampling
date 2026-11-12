"""UniPC counterpart of the team lead's Nov-2026 ablation (results/rtx3070_laptop_dpmpp_ablation_2026_11).

Protocol, same as the lead's ablation unless noted:
- gamma validation on 128 seeds (20000-20127), pixel MSE to a 100-step 2nd-order DPM-Solver++
  default-grid reference; candidates: default grid + log-SNR power gamma in {0.65, 0.8, 1.0, 1.2, 1.4};
  the gamma is chosen among the log-SNR candidates only.
- Differences: NFE 5/10/20 instead of 10 only; gamma is selected separately for every solver variant
  (the lead reused the order-2 gamma for order 1); MSE is also reported against a 1000-step reference
  on every integer timestep, to check whether the default-grid reference favours the default grid.
- Variants: UniPC order 2 / order 1 / order 2 without corrector (UniP-2); DPM-Solver++ order 2 / order 1
  rerun on the same A100 so every row of the comparison shares hardware and seeds.
- Test: FID on 1,000 seeds (10000-10999, comparable with the lead's table) and 5,000 seeds
  (1000-5999, comparable with results/a100_cn4), clean-fid CIFAR-10 train 32x32.

Run from the repository root:
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/run_ablation_unipc.py --phase all
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / 'shared_framework'
sys.path.insert(0, str(FW))
from sampler import load_pipeline, make_scheduler, sample_batch, set_custom_timesteps  # noqa: E402

MODEL_DIR = FW / 'checkpoints' / 'google-ddpm-cifar10-32'
OUT = ROOT / 'outputs' / 'ablation_unipc_2026_11'
DEVICE = 'cuda'
BATCH = 32
VAL_SEEDS = list(range(20000, 20128))
TEST_SETS = {'1k': list(range(10000, 11000)), '5k': list(range(1000, 6000))}
NFES = [5, 10, 20]
GAMMAS = [0.65, 0.8, 1.0, 1.2, 1.4]
VARIANTS = {  # name -> (solver, order, corrector)
    'unipc_o2': ('unipc', 2, True),
    'unipc_o1': ('unipc', 1, True),
    'unip_o2_nocorr': ('unipc', 2, False),
    'dpmpp_o2': ('dpmpp', 2, None),
    'dpmpp_o1': ('dpmpp', 1, None),
}


def scheduler(pipe, variant):
    solver, order, corrector = VARIANTS[variant]
    s = make_scheduler(pipe, solver, solver_order=order)
    if corrector is False:  # disable_corrector lists step indices; cover all of them
        s = UniPCMultistepScheduler.from_config(s.config, disable_corrector=list(range(1000)))
    return s


def to_images(x):
    x = (x / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(np.uint8(np.round(im * 255))) for im in x]


def sample_on_timesteps(pipe, sched, seeds, ts):
    """sample_batch with an explicit timestep list (used for the 1000-step reference)."""
    set_custom_timesteps(sched, ts, DEVICE)
    g = [torch.Generator(device='cpu').manual_seed(int(s)) for s in seeds]
    x = randn_tensor((len(seeds), 3, 32, 32), generator=g, device=torch.device(DEVICE), dtype=torch.float32)
    x = x * sched.init_noise_sigma
    with torch.inference_mode():
        for t in sched.timesteps:
            x = sched.step(pipe.unet(sched.scale_model_input(x, t), t).sample, t, x).prev_sample
    return to_images(x)


def timed(pipe, variant, seeds, nfe, gamma):
    sched = scheduler(pipe, variant)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    imgs = sample_batch(pipe, sched, seeds, nfe, gamma, DEVICE)
    torch.cuda.synchronize()
    return imgs, time.perf_counter() - t0


def arr(imgs):
    return np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])


def batches(seeds):
    for i in range(0, len(seeds), BATCH):
        yield seeds[i:i + BATCH]


def validate(pipe):
    refs = {'ref100_default': [], 'ref1000_all_t': []}
    for b in batches(VAL_SEEDS):
        refs['ref100_default'] += sample_batch(pipe, scheduler(pipe, 'dpmpp_o2'), b, 100, None, DEVICE)
        refs['ref1000_all_t'] += sample_on_timesteps(pipe, scheduler(pipe, 'dpmpp_o2'), b, list(range(999, -1, -1)))
    refs = {k: arr(v) for k, v in refs.items()}
    rows, choice = [], {}
    for v in VARIANTS:
        for nfe in NFES:
            best = (float('inf'), None)
            for gamma in [None] + GAMMAS:
                imgs = []
                for b in batches(VAL_SEEDS):
                    imgs += sample_batch(pipe, scheduler(pipe, v), b, nfe, gamma, DEVICE)
                a = arr(imgs)
                mse = {k: float(((a - r) ** 2).mean()) for k, r in refs.items()}
                rows.append(dict(variant=v, nfe=nfe, schedule='default' if gamma is None else 'logsnr_power',
                                 gamma=gamma, n_validation=len(VAL_SEEDS), **{f'mse_{k}': m for k, m in mse.items()}))
                if gamma is not None and mse['ref100_default'] < best[0]:
                    best = (mse['ref100_default'], gamma)
                print(json.dumps(rows[-1]), flush=True)
            choice[f'{v}_nfe{nfe}'] = {'gamma': best[1], 'val_mse_ref100_default': best[0]}
    write_csv(OUT / 'metrics' / 'gamma_validation.csv', rows)
    (OUT / 'metrics' / 'selected_gamma.json').write_text(json.dumps(choice, indent=2), encoding='utf-8')


def generate(pipe):
    choice = json.loads((OUT / 'metrics' / 'selected_gamma.json').read_text(encoding='utf-8'))
    timed(pipe, 'unipc_o2', [999], 5, None)  # warm-up
    rows = []
    for v in VARIANTS:
        for nfe in NFES:
            for sched_name, gamma in (('default', None), ('logsnr_power', choice[f'{v}_nfe{nfe}']['gamma'])):
                for set_name, seeds in TEST_SETS.items():
                    dest = OUT / 'samples' / set_name / f'{v}_nfe{nfe}_{sched_name}'
                    dest.mkdir(parents=True, exist_ok=True)
                    total = 0.0
                    for b in batches(seeds):
                        imgs, sec = timed(pipe, v, b, nfe, gamma)
                        total += sec
                        for im, s in zip(imgs, b):
                            im.save(dest / f'seed_{s:05d}.png')
                    rows.append(dict(variant=v, nfe=nfe, schedule=sched_name, gamma=gamma, test_set=set_name,
                                     n_samples=len(seeds), seconds=total, ms_per_image=1000 * total / len(seeds)))
                    print(json.dumps(rows[-1]), flush=True)
    write_csv(OUT / 'metrics' / 'timing.csv', rows)


def fid_phase():
    from cleanfid import fid
    ref_mu, ref_sigma = fid.get_reference_statistics('cifar10', 32, mode='clean', model_name='inception_v3', split='train')
    model = fid.build_feature_extractor('clean', torch.device(DEVICE), use_dataparallel=False)
    rows = []
    for set_name, seeds in TEST_SETS.items():
        for d in sorted((OUT / 'samples' / set_name).iterdir()):
            files = sorted(str(p) for p in d.glob('seed_*.png'))
            assert len(files) == len(seeds), (d, len(files))
            feats = fid.get_files_features(files, model=model, num_workers=0, batch_size=200, device=torch.device(DEVICE),
                                           mode='clean', description=f'{set_name}/{d.name} ', fdir=str(d), verbose=False)
            v, rest = d.name.split('_nfe')
            nfe, sched_name = rest.split('_', 1)
            rows.append(dict(variant=v, nfe=int(nfe), schedule=sched_name, test_set=set_name, n_fake=len(files),
                             fid_clean=float(fid.frechet_distance(feats.mean(0), np.cov(feats, rowvar=False), ref_mu, ref_sigma)),
                             reference='clean-fid CIFAR-10 train 32x32'))
            print(json.dumps(rows[-1]), flush=True)
    write_csv(OUT / 'metrics' / 'fid.csv', rows)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['validate', 'generate', 'fid', 'all'], default='all')
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    pipe = load_pipeline(MODEL_DIR, DEVICE) if a.phase != 'fid' else None
    if a.phase in ('validate', 'all'): validate(pipe)
    if a.phase in ('generate', 'all'): generate(pipe)
    if a.phase in ('fid', 'all'): fid_phase()


if __name__ == '__main__':
    main()
