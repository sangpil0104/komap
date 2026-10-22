"""Verify snapshot, export, and pretrained cache readiness without training."""
import gc,os,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,read_json,write_json,sha,environment
import torch
import segmentation_models_pytorch as smp
from run import snapshot
from finalize import export_model
from tqdm import tqdm

torch.set_num_threads(2)
for encoder in tqdm(['efficientnet-b4','tu-hrnet_w48'],desc='CPU pretrained cache load',unit='encoder'):
    cls=smp.Unet if encoder=='efficientnet-b4' else smp.UnetPlusPlus
    model=cls(encoder_name=encoder,encoder_weights='imagenet',in_channels=3,classes=4)
    del model; gc.collect()
with tempfile.TemporaryDirectory(prefix='runtime_check_',dir=ROOT/'checks') as temporary:
    work=Path(temporary)
    run=work/'run';run.mkdir();snapshot(run)
    manifest=read_json(run/'implementation_manifest.json')
    assert all(sha(run/'implementation'/p)==d for p,d in manifest.items())
    env=environment();env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([sys.executable,'-B',str(run/'implementation/run.py'),'check'],env=env,check=True)
    # Use a real FP16 file as the replacement, without learning or inference.
    row=read_json(ROOT/'local_weights.json')['checkpoints'][0]
    replacement=work/'replacement.pt'
    import shutil
    shutil.copy2(ROOT/row['relative'],replacement)
    export_model({'109':dict(id='synthetic_export_only',replacements={row['relative']:str(replacement)})},work/'export',0.0)
    exported=read_json(work/'export/selected_weights.json')
    assert len(list((work/'export/weights').rglob('*.pt')))==12
    assert all(sha(work/'export'/r['path'])==r['sha256'] for r in exported['checkpoints'])
    assert next(r for r in exported['checkpoints'] if r['path']==row['relative'])['source']==str(replacement)
    for path in (ROOT/'frozen/configs').rglob('*'):
        if path.is_file(): assert sha(path)==sha(work/'export/configs'/path.relative_to(ROOT/'frozen/configs'))
    assert sha(work/'export/infer_lite.py')==sha(ROOT/'upstream/infer_lite.py')
    subprocess.run([sys.executable,'-B',str(work/'export/infer_lite.py'),'--help'],env=env,stdout=subprocess.DEVNULL,check=True)
write_json(ROOT/'checks/runtime_cpu.json',dict(snapshot_verified=True,exported_weights_verified=12,
    pretrained_encoders_loaded=['efficientnet-b4','tu-hrnet_w48'],forward_calls=0,training_started=False,gpu_used=False))
print('PASS: snapshot preflight, twelve-weight export, and offline pretrained encoders; no training.')
