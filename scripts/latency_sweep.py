"""Paired service-time sensitivity, not a hardware speed benchmark."""
from pathlib import Path
import argparse
import json
import torch
from isac_wm.training import load_model
from isac_wm.evaluation import evaluate
from isac_wm.runtime import RuntimeConfig
from isac_wm.viewer import write_viewer

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--episodes', type=int, default=4)
args = parser.parse_args()
torch.set_num_threads(2)
model, scene, device = load_model(args.checkpoint, 'cpu')
summary = {'note': 'Fixed hypothetical worker service time. Publication rounds up to dt; same held-out seeds. Not a hardware benchmark or new-model accuracy claim.',
           'dt_s': scene.dt, 'episodes': args.episodes, 'seed': 50_000_000, 'profiles': {}}
for ms in (0, 50, 150, 300):
    path = args.out / f'fixed-{ms}ms'
    report = evaluate(path, model, scene, device, args.episodes, 50_000_000,
                      policies=('fixed', 'guarded'), runtime=RuntimeConfig('fixed', ms))
    write_viewer(report, path / 'replay.html')
    summary['profiles'][str(ms)] = report['summary']
    print(ms, {k: round(v['position_rmse_m'], 4) for k, v in report['summary'].items()}, flush=True)
(args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
