"""Measure this exact small model on the selected machine, not peak chip performance."""
import argparse
import json
import time
import numpy as np
import torch
from isac_wm.model import EventBeliefModel, ModelConfig, training_loss
from isac_wm.simulator import SceneConfig, recorded_episode
from isac_wm.training import batchify, resolve_device, runtime_info, seed_all
from isac_wm.evaluation import synchronize

p=argparse.ArgumentParser()
p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
p.add_argument('--iterations',type=int,default=20)
p.add_argument('--batch-size',type=int,default=4)
p.add_argument('--hidden',type=int,default=96)
p.add_argument('--features',type=int,default=48)
a=p.parse_args()
if min(a.iterations,a.batch_size)<1:p.error('positive counts required')
torch.set_num_threads(2);seed_all(1);device=resolve_device(a.device)
model=EventBeliefModel(ModelConfig(a.features,a.hidden)).to(device)
batch=batchify([recorded_episode(i,SceneConfig(steps=16)) for i in range(a.batch_size)],device)
opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
times=[]
for i in range(a.iterations+3):
    synchronize(device);t=time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss,_=training_loss(model(batch),batch)
    loss.backward();opt.step();synchronize(device)
    if i>=3:times.append((time.perf_counter()-t)*1000)
result={'runtime':runtime_info(device),'parameters':sum(p.numel() for p in model.parameters()),
        'sequence_frames':16,'image_shape':[3,24,24],'batch_size':a.batch_size,
        'warmup_iterations':3,'iterations':a.iterations,
        'train_step_p50_ms':float(np.quantile(times,.5)),
        'train_step_p95_ms':float(np.quantile(times,.95)),
        'scope':'small-model forward+backward+optimizer, not a peak hardware benchmark'}
if device.type=='cuda':result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
if device.type=='mps':
    result['mps_driver_allocated_bytes']=torch.mps.driver_allocated_memory()
    result['mps_recommended_max_memory_bytes']=torch.mps.recommended_max_memory()
print(json.dumps(result,indent=2))
