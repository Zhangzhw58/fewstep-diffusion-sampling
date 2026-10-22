from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
FW=ROOT/'shared_framework'
sys.path.insert(0,str(FW))
from sampler import load_pipeline, make_scheduler, sample_batch
MODEL_DIR=FW/'checkpoints'/'google-ddpm-cifar10-32'
OUT=ROOT/'outputs'
CONFIG=json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'))
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


def save_samples(images, seeds, target):
    target=Path(target); target.mkdir(parents=True,exist_ok=True)
    for image,seed in zip(images,seeds):
        image.save(target/f'seed_{int(seed):05d}.png')


def load():
    if DEVICE=='cuda': torch.backends.cudnn.benchmark=False
    return load_pipeline(MODEL_DIR,DEVICE)


def scheduler_for(pipe):
    return make_scheduler(pipe,'dpmpp',solver_order=2)


def timed_sample(pipe, seeds, n_steps, gamma=None):
    sched=scheduler_for(pipe)
    if DEVICE=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    images=sample_batch(pipe,sched,seeds,n_steps,gamma,DEVICE)
    if DEVICE=='cuda': torch.cuda.synchronize()
    return images,time.perf_counter()-start


def smoke():
    pipe=load(); seeds=[CONFIG['smoke_seed']]
    imgs,elapsed=timed_sample(pipe,seeds,5,None)
    save_samples(imgs,seeds,OUT/'samples'/'smoke_dpmpp_5step')
    print(json.dumps({'device':DEVICE,'model':CONFIG['model_id'],'nfe':5,'seconds':elapsed,'size':imgs[0].size},ensure_ascii=False))


def validation():
    pipe=load(); val_seeds=CONFIG['validation_seeds']; steps_list=CONFIG['n_steps']; gammas=CONFIG['gamma_candidates']
    ref_dir=OUT/'validation'/'reference_100step'
    refs={}
    for start in range(0,len(val_seeds),CONFIG['batch_size']):
        batch=val_seeds[start:start+CONFIG['batch_size']]
        imgs,sec=timed_sample(pipe,batch,CONFIG['reference_steps'],None)
        save_samples(imgs,batch,ref_dir)
        refs.update({seed:img for seed,img in zip(batch,imgs)})
    choices={}
    rows=[]
    for n in steps_list:
        best=(float('inf'),None)
        for gamma in gammas:
            all_imgs=[]
            for start in range(0,len(val_seeds),CONFIG['batch_size']):
                batch=val_seeds[start:start+CONFIG['batch_size']]
                imgs,sec=timed_sample(pipe,batch,n,float(gamma))
                save_samples(imgs,batch,OUT/'validation'/f'n{n}'/f'gamma_{gamma:g}')
                all_imgs.extend(imgs)
            mse=[]
            for seed,img in zip(val_seeds,all_imgs):
                a=np.asarray(img,dtype=np.float32)/255.0
                b=np.asarray(refs[seed],dtype=np.float32)/255.0
                mse.append(float(np.mean((a-b)**2)))
            score=float(np.mean(mse))
            rows.append({'n_steps':n,'gamma':gamma,'val_mse_to_reference':score})
            if score<best[0]: best=(score,float(gamma))
        choices[str(n)]={'gamma':best[1],'validation_mse':best[0]}
    (OUT/'metrics').mkdir(parents=True,exist_ok=True)
    (OUT/'metrics'/'selected_gamma.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    with (OUT/'metrics'/'validation_search.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps(choices,indent=2))


def prepare_real():
    ds=CIFAR10(root=str(ROOT/'data'/'cifar10'),train=False,download=True)
    n=int(CONFIG['n_test_samples'])
    dest=OUT/'real_cifar10_test'
    dest.mkdir(parents=True,exist_ok=True)
    for i in range(min(n,len(ds))):
        img,_=ds[i]; img.save(dest/f'real_{i:05d}.png')
    print(f'saved {min(n,len(ds))} CIFAR-10 test images to {dest}')


def generate():
    pipe=load(); choices=json.loads((OUT/'metrics'/'selected_gamma.json').read_text(encoding='utf-8'))
    seeds=list(range(CONFIG['test_seed_start'],CONFIG['test_seed_start']+CONFIG['n_test_samples']))
    summary=[]
    # Warm-up is measured separately and not included in per-config timing.
    timed_sample(pipe,[CONFIG['warmup_seed']],5,None)
    for n in CONFIG['n_steps']:
        for name,gamma in [('default',None),('logsnr_power',float(choices[str(n)]['gamma']))]:
            dest=OUT/'samples'/f'dpmpp_n{n}_{name}'
            durations=[]
            for start in range(0,len(seeds),CONFIG['batch_size']):
                batch=seeds[start:start+CONFIG['batch_size']]
                imgs,sec=timed_sample(pipe,batch,n,gamma)
                save_samples(imgs,batch,dest)
                durations.append(sec)
                print(f'{n=} {name=} {start+len(batch)}/{len(seeds)} batch_sec={sec:.2f}',flush=True)
            summary.append({'method':'DPM-Solver++','n_steps':n,'schedule':name,'gamma':gamma,
                            'n_samples':len(seeds),'total_sampling_sec':sum(durations),
                            'seconds_per_image':sum(durations)/len(seeds),'device':DEVICE,
                            'batch_size':CONFIG['batch_size']})
    with (OUT/'metrics'/'timing.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    print(json.dumps(summary,indent=2))


def calculate_fid():
    from cleanfid import fid
    # clean-fid's glob patterns include both *.png and *.PNG. Windows glob is
    # case-insensitive, so those patterns double-count every PNG. Build a
    # unique file list ourselves and call the feature API directly.
    ref_mu, ref_sigma = fid.get_reference_statistics(
        'cifar10', 32, mode='clean', model_name='inception_v3', split='train'
    )
    feat_model = fid.build_feature_extractor(
        'clean', torch.device(DEVICE), use_dataparallel=False
    )
    rows=[]
    for n in CONFIG['n_steps']:
        for name in ['default','logsnr_power']:
            fake=OUT/'samples'/f'dpmpp_n{n}_{name}'
            files=sorted(p for p in fake.iterdir() if p.is_file() and p.suffix.lower() in
                         {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff','.ppm','.pgm'})
            if len(files) != int(CONFIG['n_test_samples']):
                raise RuntimeError(f'Expected {CONFIG["n_test_samples"]} unique images in {fake}, found {len(files)}')
            print(f'Extracting {len(files)} unique images from {fake}',flush=True)
            feats=fid.get_files_features([str(p) for p in files], model=feat_model,
                                         num_workers=0,batch_size=32,device=torch.device(DEVICE),
                                         mode='clean',description=f'FID {fake.name}: ',
                                         fdir=str(fake),verbose=True)
            mu=np.mean(feats,axis=0); sigma=np.cov(feats,rowvar=False)
            value=fid.frechet_distance(mu,sigma,ref_mu,ref_sigma)
            rows.append({'method':'DPM-Solver++','n_steps':n,'schedule':name,'fid_clean':float(value),
                         'n_fake':CONFIG['n_test_samples'],
                         'reference_stats':'clean-fid CIFAR-10 train 32x32'})
            print(rows[-1],flush=True)
    with (OUT/'metrics'/'fid.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['smoke','validate','generate','fid','all'],default='smoke')
    args=p.parse_args()
    phases=['smoke','validate','generate','fid'] if args.phase=='all' else [args.phase]
    for phase in phases:
        print(f'=== {phase} ===',flush=True)
        {'smoke':smoke,'validate':validation,'generate':generate,'fid':calculate_fid}[phase]()

if __name__=='__main__': main()

