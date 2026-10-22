from pathlib import Path
import ast,os,sys,subprocess
from tqdm import tqdm
from common import ROOT,CONDA,GPUS,read,sha,devices,now
from prediction_checks import validate_prediction_inventory

def check():
    if Path(sys.prefix).resolve()!=CONDA.resolve():raise RuntimeError('Use conda komap through run.sh')
    for name,digest in tqdm(read(ROOT/'input_manifest.json').items(),desc='Verify preserved inputs',unit='file',mininterval=5):
        if sha(name)!=digest:raise ValueError(f'Parent/input changed: {name}')
    for name,digest in read(ROOT/'code_manifest.json').items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Resume code changed: {name}')
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(),filename=name)
    parent=Path(read(ROOT/'parent.json')['parent_run'])
    for p in (ROOT/'source').rglob('*'):
        if p.is_file() and p.read_bytes()!=(parent/'implementation/source'/p.relative_to(ROOT/'source')).read_bytes():raise ValueError(f'Inference/training source changed: {p}')
    validate_prediction_inventory(ROOT/'data/valid/images',parent/'pred_valid',20)
    audit=read(ROOT/'checks/parent_audit.json')
    if not audit['ok'] or audit['completed_training']!=33 or len(audit['deployed_weights'])!=13:raise ValueError('Invalid parent audit')
    hardware=devices();os.environ['CUDA_VISIBLE_DEVICES']=','.join(hardware[i]['uuid'] for i in GPUS)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4:raise RuntimeError('Expected CUDA GPUs 3-6')
    for p in ROOT.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    return dict(ok=True,checked_at=now(),parent_run=str(parent),reused_training_jobs=33,new_training_jobs=0,reused_weights=13,
                resume_steps=['valid inference + measured timing','test inference + measured timing','valid scoring + report export'],
                physical_gpus=GPUS,hardware=hardware,python=sys.executable,training_started=False,inference_started=False)
