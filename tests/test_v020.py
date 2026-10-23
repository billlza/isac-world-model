import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from isac_wm.events import Observation, EventQueue, SnapshotBuffer
from isac_wm.baselines import ReferenceReplayKalman, ReplayKalman, risk_budget_action, AirtimeBudget
from isac_wm.simulator import SceneConfig, ToyScene, recorded_episode
from isac_wm.scenarios import scenario_config, SCENARIOS
from isac_wm.model import EventBeliefModel, ModelConfig, training_loss
from isac_wm.training import batchify, seed_all, train, load_model
from isac_wm.calibration import fit_episode_quantile, validate_calibration, load_calibration
from isac_wm.artifacts import save_checkpoint, read_checkpoint, write_json, model_fingerprint
from isac_wm.safety import MonitorConfig, assess
from isac_wm.evaluation import evaluate, rollout


def radio(i=1, capture=0., arrival=.1):
    return Observation(i, 'radio', capture, arrival, np.array([.4, 1., 0., 0.]), 1)


@pytest.mark.parametrize('uid', [-1, True, 1.2])
def test_bad_observation_ids(uid):
    with pytest.raises(ValueError): radio(uid)


@pytest.mark.parametrize('payload', [np.zeros(3), np.zeros((4,1)), np.full(4, np.nan)])
def test_bad_payload(payload):
    with pytest.raises(ValueError): Observation(1,'radio',0.,.1,payload,1)


def test_observation_owns_readonly_copy():
    a=np.ones(4, dtype=np.float32)
    o=Observation(1,'radio',0.,0.,a,1);a[0]=99
    assert o.payload[0]==1 and not o.payload.flags.writeable


def test_queue_total_id_budget():
    q=EventQueue(capacity=1,max_ids=1);q.push(radio());q.release(.1)
    with pytest.raises(OverflowError):q.push(radio(2,.2,.3))


def test_snapshot_atomic_rejection_and_duplicates():
    b=SnapshotBuffer(24)
    with pytest.raises(ValueError):b.update([radio(1),radio(2,0.,2.)],.1,.1)
    assert b.latest=={} and len(b._seen)==0
    b.update([radio()],.1,.1)
    with pytest.raises(ValueError):b.update([radio()],.2,.1)
    with pytest.raises(ValueError):b.update([],0.,.1)


def test_snapshot_dimension_guard():
    b=SnapshotBuffer(24)
    o=Observation(2,'vision',0.,0.,np.zeros((3,16,16)),0)
    with pytest.raises(ValueError):b.update([o],0.,.1)


@pytest.mark.parametrize('n', [12,48,120])
def test_cached_replay_equals_reference_with_oosm(n):
    c=SceneConfig(steps=n,delay_jitter_s=.18)
    scene=ToyScene(532,c);q=EventQueue();a=ReferenceReplayKalman(c.room_m);b=ReplayKalman(c.room_m)
    for k in range(n):
        now=k*c.dt;ready=q.release(now)
        a.ingest(ready,now);b.ingest(ready,now)
        x,p=a.estimate(now);y,r=b.estimate(now)
        np.testing.assert_allclose(x,y,rtol=2e-10,atol=2e-10)
        np.testing.assert_allclose(p,r,rtol=2e-10,atol=2e-10)
        before=b.replayed_updates;b.estimate(now)
        assert before==b.replayed_updates
        for o in (scene.camera(k),scene.radio(k,1 if k%3 else 2)):
            if o is not None:q.push(o)
    assert b.replayed_updates < n*n


def test_replay_ingest_atomic():
    k=ReplayKalman(10)
    with pytest.raises(ValueError):k.ingest([radio(),radio(2,0.,1.)],.1)
    assert not k.observations
    k.ingest([radio()],.1)
    with pytest.raises(ValueError):k.estimate(0.)


def test_gating_is_opt_in_and_counted():
    a=ReplayKalman(10,gate_threshold=.01)
    a.ingest([radio()],.1);a.estimate(.1)
    assert a.rejected_measurements==1


def test_batched_encoder_matches_sequential_and_gradients():
    torch.set_num_threads(2);seed_all(1)
    c=SceneConfig(steps=12)
    b=batchify([recorded_episode(i,c) for i in (3,4)],torch.device('cpu'))
    a=EventBeliefModel(ModelConfig(16,32));other=copy.deepcopy(a)
    o=a(b,batched_encoding=True);p=other(b,batched_encoding=False)
    for key in o:torch.testing.assert_close(o[key],p[key],rtol=2e-5,atol=2e-6)
    training_loss(o,b)[0].backward();training_loss(p,b)[0].backward()
    for x,y in zip(a.parameters(),other.parameters()):
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=2e-4,atol=3e-6)


@pytest.mark.parametrize('h', [float('nan'),float('inf'),-1.])
def test_bad_forecast_horizons(h):
    m=EventBeliefModel(ModelConfig(16,32))
    with pytest.raises(ValueError):m.forecast(torch.zeros(1,32),h)


def test_artifact_checksum_and_strict_json(tmp_path):
    p=tmp_path/'model.pt';save_checkpoint(p,{'schema_version':2,'x':torch.ones(2)})
    assert read_checkpoint(p)['x'].sum()==2
    with p.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='SHA'):read_checkpoint(p)
    with pytest.raises(ValueError):write_json(tmp_path/'bad.json',{'v':float('nan')})
    assert not (tmp_path/'bad.json').exists()


def test_checkpoint_schema_and_unsafe_pickle(tmp_path):
    p=tmp_path/'bad.pt';save_checkpoint(p,{'schema_version':99})
    with pytest.raises(ValueError,match='schema'):read_checkpoint(p)
    p=tmp_path/'notzip.pt';p.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='ZIP'):read_checkpoint(p)


def test_symlink_output_rejected(tmp_path):
    target=tmp_path/'target';target.write_text('keep');link=tmp_path/'link';link.symlink_to(target)
    with pytest.raises(ValueError):write_json(link,{'bad':True})
    assert target.read_text()=='keep'


def dummy_runs(n=10):
    return [{'seed':i,'trace':[{'estimate':[float(i+1),0,0,0], 'truth':[0,0,0,0],
                               'position_variance':[1,1]} for _ in range(6)]} for i in range(n)]


def test_conformal_uses_episode_max_and_finite_sample_rank():
    runs=dummy_runs(10)
    q=fit_episode_quantile(runs,alpha=.1)
    assert q['rank']==10 and q['quantile']==10 and q['n_episodes']==10
    with pytest.raises(ValueError):fit_episode_quantile(dummy_runs(2),alpha=.1)
    runs[1]['seed']=runs[0]['seed']
    with pytest.raises(ValueError):fit_episode_quantile(runs)


def test_calibration_model_scene_and_split_guards():
    c=SceneConfig();m=EventBeliefModel(ModelConfig(16,32))
    d={'scene':c.to_dict(),'model_fingerprint':model_fingerprint(m),
       'variants':{'model+fixed':{'episode_seeds':[1,2]}}}
    validate_calibration(d,m,c,[3])
    with pytest.raises(ValueError,match='overlap'):validate_calibration(d,m,c,[2])
    with pytest.raises(ValueError,match='mismatch'):validate_calibration(d,m,scenario_config('delay',c),[3])
    with torch.no_grad():next(m.parameters()).add_(.1)
    with pytest.raises(ValueError,match='mismatch'):validate_calibration(d,m,c,[3])


def test_invalid_calibration_file(tmp_path):
    p=tmp_path/'c.json';p.write_text(json.dumps({'schema_version':1,'variants':{'m':{'quantile':-1}}}))
    with pytest.raises(ValueError):load_calibration(p)


def verdict(**kw):
    values=dict(state=np.array([2.,2.,0.,0.]),variance=np.full(4,.01),valid=np.ones(2),
                age=np.array([.1,.1]),compute_ms=1.,room_m=10.,config=MonitorConfig())
    values.update(kw);return assess(**values)


def test_no_acceptance_without_calibration():
    assert verdict()['status']=='uncalibrated' and not verdict()['accepted']
    assert verdict(quantile=2.)['accepted']


@pytest.mark.parametrize('kw,status', [
    ({'valid':np.zeros(2)},'no_recent_evidence'),
    ({'age':np.ones(2)},'no_recent_evidence'),
    ({'compute_ms':200},'compute_deadline_missed'),
    ({'compute_ms':-1},'invalid_numeric'),
    ({'valid':np.array([2,0])},'invalid_numeric'),
    ({'state':np.array([np.nan,0,0,0])},'invalid_numeric'),
    ({'state':np.array([-1,0,0,0])},'outside_model_domain'),
    ({'variance':np.ones(4)},'uncertain')])
def test_result_monitor_boundaries(kw,status):
    r=verdict(quantile=2.,**kw);assert r['status']==status and not r['accepted']


def test_faults_do_not_change_hidden_trajectory():
    c=SceneConfig();base=ToyScene(3,c).truth_for_evaluation()
    for name in SCENARIOS[:-1]:
        np.testing.assert_array_equal(base,ToyScene(3,scenario_config(name,c)).truth_for_evaluation())
    scene=ToyScene(4,scenario_config('blackout',c))
    assert all(scene.camera(i) is None and scene.radio(i,1) is None for i in range(c.steps))


def test_risk_policy_can_skip_and_request_precise():
    assert risk_budget_action(np.array([.001,.001]),np.zeros(2),.6,10.)==0
    assert risk_budget_action(np.array([2.,2.]),np.zeros(2),.6,10.,target_radius_m=.2)==2
    assert risk_budget_action(np.array([2.,2.]),np.zeros(2),0.,10.)==0


def test_random_risk_policy_respects_budget():
    rng=np.random.default_rng(14);b=AirtimeBudget(.1)
    for k in range(200):
        b.refill();a=risk_budget_action(rng.random(2),rng.random(2),b.tokens,10.);b.authorize(a)
        assert b.spent <= .1*(k+1)+1e-9


def test_resume_cpu_reproduces_uninterrupted_training(tmp_path):
    torch.set_num_threads(2)
    common=dict(episodes=4,batch_size=2,device_name='cpu',seed=3,
                scene=SceneConfig(steps=8),model_config=ModelConfig(16,32))
    train(tmp_path/'full',epochs=2,**common)
    train(tmp_path/'part',epochs=1,**common)
    train(tmp_path/'resume',epochs=2,resume=tmp_path/'part/resume.pt',**common)
    a=read_checkpoint(tmp_path/'full/resume.pt');b=read_checkpoint(tmp_path/'resume/resume.pt')
    for k in a['state_dict']:torch.testing.assert_close(a['state_dict'][k],b['state_dict'][k],rtol=0,atol=0)
    assert a['history']==b['history']
    with pytest.raises(ValueError,match='match'):
        train(tmp_path/'wrong',epochs=2,resume=tmp_path/'part/resume.pt',**{**common,'seed':4})


def test_memory_budget_prevents_dataset_allocation(tmp_path):
    with pytest.raises(MemoryError):train(tmp_path/'limited',epochs=1,episodes=4,max_dataset_bytes=1)


def test_evaluation_rejects_train_seed_overlap(tmp_path):
    m=EventBeliefModel(ModelConfig(16,32));m.provenance_seeds={1}
    with pytest.raises(ValueError,match='overlap'):evaluate(tmp_path,m,SceneConfig(steps=8),torch.device('cpu'),episodes=1,seed=1)


def test_blackout_is_not_silently_accepted():
    r=rollout(42,scenario_config('blackout',SceneConfig(steps=10)),'kalman','risk',quantile=1.)
    assert r['metrics']['acceptance_fraction']==0
    assert all(x['status']=='no_recent_evidence' for x in r['trace'])


def test_guard_requires_fresh_evidence_even_when_variance_is_tiny():
    from isac_wm.baselines import guarded_risk_action
    p=np.full(2,1e-6);v=np.zeros(2)
    assert guarded_risk_action(p,v,.1,10,np.ones(2),np.array([.1,.4]))==1
    assert guarded_risk_action(p,v,.1,10,np.ones(2),np.array([.1,.1]))==0
    assert guarded_risk_action(p,v,.1,10,np.zeros(2),np.zeros(2))==1
    assert guarded_risk_action(p,v,0,10,np.zeros(2),np.zeros(2))==0


def test_viewer_escapes_script_and_rejects_empty(tmp_path):
    from isac_wm.viewer import write_viewer
    report={"runs":[{"trace":[{"x":"</script><script>bad()</script>"}]}]}
    path=tmp_path/'viewer.html';write_viewer(report,path)
    text=path.read_text()
    assert "</script><script>bad" not in text and "requestAnimationFrame" in text
    assert "setInterval" not in text
    with pytest.raises(ValueError):write_viewer({"runs":[]},tmp_path/'empty.html')


@pytest.mark.parametrize('value', [[],None,{}, {'schema_version':1,'variants':{}}])
def test_calibration_schema_rejects_incomplete_documents(tmp_path,value):
    p=tmp_path/'invalid.json';p.write_text(json.dumps(value))
    with pytest.raises(ValueError):load_calibration(p)
