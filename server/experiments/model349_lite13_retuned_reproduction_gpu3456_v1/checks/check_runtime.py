"""CPU imports/argument parsing and offline ImageNet initialization, without forward/backward."""
from pathlib import Path
import gc, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common import read, write
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
rows=read(ROOT/'plan.json')['tasks'];seen=set();checks=[]
for row in rows:
    command=row['command'];script=(Path(row['cwd'])/command[3]).resolve()
    # Check each distinct full argument schema, including each HR input route.
    key=str(script)
    if key in seen:continue
    seen.add(key);env=os.environ.copy()
    for k,v in row['env'].items():
        if v is not None and k!='CUDA_VISIBLE_DEVICES':env[k]=v
    result=subprocess.run([*command,'--help'],cwd=row['cwd'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (ROOT/'checks'/f'help_{row["name"]}.log').write_text(result.stdout)
    if result.returncode:raise RuntimeError(f'Help failed: {row["name"]}\n{result.stdout[-6000:]}')
    checks.append(row['name']);print('HELP PASS',row['name'],flush=True)
sys.path[:0]=[str(ROOT/'source/training/train_route/experiments'),str(ROOT/'source/training/train_route/experiments/historical_scripts')]
import torch
import segmentation_models_pytorch as smp
from train_models_2_to_6 import UNetPlusPlusResNet34
torch.set_num_threads(2)
for name in ['resnet34','efficientnet-b4','tu-hrnet_w48']:
    if name=='resnet34':model=UNetPlusPlusResNet34(pretrained=True)
    else:model=(smp.Unet if name=='efficientnet-b4' else smp.UnetPlusPlus)(encoder_name=name,encoder_weights='imagenet',in_channels=3,classes=4)
    if not all(torch.isfinite(v).all() for v in model.parameters()):raise ValueError('Nonfinite pretrained initialization')
    del model;gc.collect();print('CPU IMAGENET PASS',name,flush=True)
write(ROOT/'checks/runtime_cpu.json',dict(ok=True,help_checks=checks,offline_imagenet=['resnet34','efficientnet-b4','tu-hrnet_w48'],forward_calls=0,backward_calls=0,training_started=False,gpu_inference_started=False))
