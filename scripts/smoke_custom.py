from pathlib import Path
import sys,torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'shared_framework'))
from sampler import load_pipeline,make_scheduler,sample_batch,logsnr_power_timesteps
pipe=load_pipeline(ROOT/'shared_framework/checkpoints/google-ddpm-cifar10-32','cuda')
for n in [5,10,20]:
 s=make_scheduler(pipe,'dpmpp'); ts=logsnr_power_timesteps(s,n,0.8); s.set_timesteps(timesteps=ts,device='cuda'); imgs=sample_batch(pipe,make_scheduler(pipe,'dpmpp'),[321],n,0.8,'cuda'); print(n,ts,imgs[0].size)
