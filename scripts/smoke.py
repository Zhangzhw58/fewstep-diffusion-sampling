import os, time, json
from pathlib import Path
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'models'/'hf_cache')
MODEL='google/ddpm-cifar10-32'
print('torch',torch.__version__,'cuda',torch.cuda.is_available(), flush=True)
print('loading',MODEL,flush=True)
pipe=DiffusionPipeline.from_pretrained(MODEL,torch_dtype=torch.float32,local_files_only=True)
print('pipeline',type(pipe).__name__,'scheduler',type(pipe.scheduler).__name__, flush=True)
pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,algorithm_type='dpmsolver++',solver_order=2)
pipe=pipe.to('cuda')
g=torch.Generator(device='cpu').manual_seed(123)
t0=time.perf_counter()
out=pipe(batch_size=1,num_inference_steps=5,generator=g)
torch.cuda.synchronize()
print('images',len(out.images),'size',out.images[0].size,'elapsed',time.perf_counter()-t0, flush=True)
path=ROOT/'outputs'/'samples'/'smoke_dpmpp_5step.png'; path.parent.mkdir(parents=True,exist_ok=True); out.images[0].save(path)
print('saved',path,flush=True)
