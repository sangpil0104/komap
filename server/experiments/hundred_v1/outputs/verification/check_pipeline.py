"""Audit smoke checkpoints and reports after distributed integration checks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
import json
import pandas as pd
import torch
from hundredlab.workspace import config,read_json,run_dir,source_hash
from hundredlab.catalog import catalog
from alphase.utils import json_hash,sha256

names=[list(catalog())[i] for i in (0,14,18,22,36,47,61,63,70,85,90,93,96,99)]
records=[]
for name in names:
    directory=run_dir(name,True);cfg=config(name,True)
    status=read_json(directory/'status.json')
    assert status['state']=='completed'
    initial=read_json(directory/'initial_score.json')['image_mIoU']
    last=torch.load(directory/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    safe=torch.load(directory/'best.pt',map_location='cpu',weights_only=False,mmap=True)
    trained=torch.load(directory/'best_trained.pt',map_location='cpu',weights_only=False,mmap=True)
    assert last['completed'] and last['world_size']==8
    assert last['optimizer_step']==sum(s['max_updates'] for s in cfg['train']['stages'])
    assert last['config_hash']==json_hash(cfg)
    assert last['resolved_config']['hundred']['source_sha256']==source_hash()
    assert safe['best_metric']>=initial
    assert trained['optimizer_step']>0
    for mode in ('plain','d4'):
        folder=ROOT/'outputs/smoke_evaluations'/name/mode
        provenance=read_json(folder/'provenance.json')
        assert provenance['request']['checkpoint_sha256']==sha256(directory/'best_trained.pt')
        assert provenance['request']['source_sha256']==source_hash()
        data=pd.read_csv(folder/'per_image_metrics.csv')
        assert len(data)==20 and data.image_id.is_unique
        if mode=='plain':assert abs(data.mIoU_exclude.mean()-trained['best_metric'])<1e-7
    records.append(dict(experiment=name,updates=last['optimizer_step'],initial_in_selection=True,plain_d4_images=20))
    del last,safe,trained
assert not (ROOT/'outputs/runs').exists()
assert not (ROOT.parents[1]/'model_results_0920.md').exists()
assert not list((ROOT/'outputs').glob('smoke_runs/**/index.html'))
assert 'SMOKE' in (ROOT/'outputs/smoke_reports/model_results_0920.md').read_text()
assert not read_json(ROOT/'outputs/smoke_reports/completion.json')['completed']
log=(ROOT/'outputs/verification/smoke_suite.log').read_text()
assert 'ETA=' in log and 'Monitor:' not in log
(ROOT/'outputs/verification/pipeline_audit.json').write_text(json.dumps(dict(
    experiments=records,source_sha256=source_hash(),real_training_started=False,
    smoke_excluded_from_final_report=True,terminal_progress_only=True),indent=2))
print('14 distributed smoke pipelines and protected initial selection: PASS')
