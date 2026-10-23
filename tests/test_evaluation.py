import json
import numpy as np
import torch
from isac_wm.evaluation import evaluate,rollout
from isac_wm.model import EventBeliefModel,ModelConfig
from isac_wm.simulator import SceneConfig
from isac_wm.viewer import write_viewer


def test_full_smoke_rollout():
    r=rollout(12,SceneConfig(steps=12),'kalman','fixed')
    assert len(r['trace'])==12
    assert r['metrics']['sensing_airtime_fraction']<=.1+1e-8
    assert r['trace'][0]['arrived_ids']==[]
    assert all(np.isfinite(row['estimate']).all() for row in r['trace'])


def test_all_four_combinations_and_viewer(tmp_path):
    torch.set_num_threads(2)
    m=EventBeliefModel(ModelConfig(16,32)).eval()
    r=evaluate(tmp_path,m,SceneConfig(steps=10),torch.device('cpu'),episodes=1)
    assert len(r['runs'])==4
    write_viewer(r,tmp_path/'replay.html')
    assert '__PAYLOAD__' not in (tmp_path/'replay.html').read_text()
    assert len(json.loads((tmp_path/'evaluation.json').read_text())['summary'])==4
