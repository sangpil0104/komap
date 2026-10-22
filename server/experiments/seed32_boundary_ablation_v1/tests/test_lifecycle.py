"""Safety and continuity of the actual suite coordinator/checkpoint format."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

import run as runner
import worker
from boundarylab.config import make_plan
from boundarylab.data import TrainDataset, one_sample_collate
from boundarylab.io import capture_rng, fingerprint, immutable_torch_save, read_json, restore_rng, seed_all, sha256, write_json
from test_protocol import fixture_rows


def test_spawn_loader_and_resume_indices(tmp_path):
    cfg=make_plan(crop=64,steps=2)[3]['config'].copy();cfg['target_width']=256
    ds=TrainDataset(fixture_rows(tmp_path),cfg)
    loader=DataLoader(ds,batch_size=1,sampler=range(3,16,8),num_workers=1,
                      multiprocessing_context='spawn',collate_fn=one_sample_collate)
    samples=list(loader)
    assert [x['sample_index'] for x in samples]==[3,11]
    assert torch.equal(samples[-1]['detail'][0],ds[11]['detail'])


def test_optimizer_scheduler_rng_roundtrip_with_synthetic_toy(tmp_path):
    # Tiny synthetic linear arithmetic only, not a B4 or dataset training run.
    seed_all(32)
    net=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(net.parameters(),lr=.003)
    cfg=dict(warmup=2,steps=5)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:worker.lr_factor(s,cfg))
    def toy_step():
        x=torch.randn(2,3)
        opt.zero_grad(set_to_none=True)
        net(x).square().mean().backward()
        opt.step();scheduler.step()
    toy_step()
    state=dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
               step=1,rng=[capture_rng()],protocol_hash='fixture',elapsed_seconds=1.0)
    path=tmp_path/'checkpoints'/'step_000001.pt'
    immutable_torch_save(state,path)
    toy_step()
    expected={k:v.detach().clone() for k,v in net.state_dict().items()}
    expected_lr=scheduler.get_last_lr()
    restored=torch.load(path,map_location='cpu',weights_only=False)
    net.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer'])
    scheduler.load_state_dict(restored['scheduler']);restore_rng(restored['rng'][0])
    toy_step()
    for k,v in net.state_dict().items():
        assert torch.equal(v,expected[k])
    assert scheduler.get_last_lr()==expected_lr
    # An interrupted write leaves a .tmp; only completed checkpoints are resumable.
    (path.parent/'step_000999.pt.tmp').write_text('incomplete')
    assert worker.latest_checkpoint(tmp_path)==path


def test_real_launcher_uses_eight_gpus_and_waits_between_trials(tmp_path,monkeypatch):
    suite=tmp_path
    (suite/'trials').mkdir();(suite/'logs').mkdir();(suite/'source').mkdir()
    plan=dict(gpus=list(range(8)))
    events=[]
    active=[False]
    class Child:
        pid=12345
        returncode=None
        def __init__(self,cmd,**kw):
            assert not active[0], 'Overlapping experiment launches'
            active[0]=True
            assert '--nproc_per_node=8' in cmd
            assert kw['env']['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
            assert kw['env']['PYTHONDONTWRITEBYTECODE']=='1'
            assert Path(cmd[-5]).name=='worker.py'
            self.arm=cmd[-1];self.polls=0
            events.append(('start',self.arm))
        def poll(self):
            self.polls+=1
            if self.polls==1:
                return None
            self.returncode=0;active[0]=False
            events.append(('end',self.arm))
            return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    for arm in make_plan()[:2]:
        runner.launch(suite,plan,arm,0)
    assert events==[('start','R00_seed32'),('end','R00_seed32'),('start','R01_no_tta'),('end','R01_no_tta')]


def test_failed_worker_is_not_silently_accepted(tmp_path,monkeypatch):
    for p in ('trials','logs','source'):(tmp_path/p).mkdir()
    class Failed:
        pid=12345
        returncode=7
        def __init__(self,*a,**kw):kw['stdout'].write('injected failure\n');kw['stdout'].flush()
        def poll(self):return 7
    monkeypatch.setattr(runner.subprocess,'Popen',Failed)
    with pytest.raises(RuntimeError,match='injected failure'):
        runner.launch(tmp_path,dict(gpus=list(range(8))),make_plan()[0],0)
    assert read_json(tmp_path/'status.json')['phase']=='failed'


def test_interrupt_terminates_only_launched_process_group(tmp_path,monkeypatch):
    for p in ('trials','logs','source'):(tmp_path/p).mkdir()
    killed=[]
    class Running:
        pid=12345
        returncode=None
        def __init__(self,*a,**kw):pass
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=-15;return -15
    monkeypatch.setattr(runner.subprocess,'Popen',Running)
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    def stop(_):runner.request_stop(15,None)
    monkeypatch.setattr(runner,'read_progress',stop)
    with pytest.raises(KeyboardInterrupt):
        runner.launch(tmp_path,dict(gpus=list(range(8))),make_plan()[0],0)
    assert killed==[(12345,runner.signal.SIGTERM)]
    assert read_json(tmp_path/'status.json')['phase']=='interrupted'


def test_resume_detects_mutated_data_and_frozen_source(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    suite=tmp_path/'outputs'/'fixture';(suite/'source').mkdir(parents=True)
    source=suite/'source'/'worker.py';source.write_text('original')
    inspected=dict(data_hash='data',encoder_sha256='encoder',reference_sha256='reference',environment={'x':1},gpus=list(range(8)))
    plan=dict(inspected,arms=make_plan(),source_files={'worker.py':sha256(source)})
    write_json(suite/'plan.json',plan)
    assert runner.validate_resume(suite,inspected)['data_hash']=='data'
    changed=dict(inspected,data_hash='changed')
    with pytest.raises(ValueError,match='data_hash'):
        runner.validate_resume(suite,changed)
    source.write_text('modified')
    with pytest.raises(ValueError,match='Frozen source changed'):
        runner.validate_resume(suite,inspected)
