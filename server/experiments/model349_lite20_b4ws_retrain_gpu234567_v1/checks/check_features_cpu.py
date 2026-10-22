"""Synthetic train-OOF and refiner-input checks, without any optimizer step."""
from pathlib import Path
import importlib.util,sys,tempfile
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT,write_json,read_json
from features import prepare,blend,PALETTE
import torch
torch.set_num_threads(2)
sys.path.insert(0,str(ROOT/'fusion_source/refiner'))
from train_residual import Crops
from gray_input import prepare_gray
from residual_model import Residual

p=np.broadcast_to(np.asarray([.1,.2,.3,.4],np.float16)[:,None,None],(4,256,256)).copy()
mixed=blend(p,p,p)
np.testing.assert_array_equal(mixed,(.5*p.astype(np.float32)+.25*p.astype(np.float32)+.25*p.astype(np.float32))[[1,2,0,3]])
gray=np.arange(256*256,dtype=np.float32).reshape(256,256)/(256*256)
np.testing.assert_allclose(prepare_gray(gray),prepare_gray(gray*2+7),atol=2e-6)
assert np.all(prepare_gray(np.zeros((256,256),np.float32))==.5)
assert 0<=prepare_gray(gray).min()<=prepare_gray(gray).max()<=1
with tempfile.TemporaryDirectory(prefix='features_',dir=ROOT/'checks') as directory:
    work=Path(directory);data=work/'data';run=work/'run'
    (data/'train/images').mkdir(parents=True);(data/'train/masks').mkdir()
    targets=np.tile(np.arange(256,dtype=np.uint8)[None,:]//64,(256,1))
    all_names=[]
    for fold in range(1,5):
        names=sorted(p.stem.removesuffix('_image') for p in (ROOT/f'expert_splits/model329_cv/oof_fold{fold}/images').glob('*.png'))
        all_names+=names
        for name in names:
            Image.fromarray((gray*255).astype(np.uint8)).save(data/f'train/images/{name}_image.png')
            Image.fromarray(PALETTE[targets]).save(data/f'train/masks/{name}_mask.png')
        for family in ['b4','hr1','hr2']:
            folder=run/f'expert_oof/{family}/fold{fold}';folder.mkdir(parents=True)
            np.savez_compressed(folder/'probabilities.npz',**{n:p for n in names})
    prepare(run,data,ROOT/'expert_splits',work/'features')
    manifest=read_json(work/'features/manifest.json')
    assert len(manifest['images'])==70 and manifest['valid_read'] is False and manifest['test_read'] is False
    dataset=Crops(work/'features');assert len(dataset)==1120
    a,b,c=dataset[0]
    assert a.shape==(4,256,256) and b.shape==(1,256,256) and c.shape==(256,256)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    model=Residual();assert sum(t.numel() for t in model.parameters())==11876
    with torch.no_grad():output,delta=model(a[None],b[None])
    assert torch.equal(output,a[None]) and torch.count_nonzero(delta)==0
    # Foreign/duplicate IDs in an OOF archive must fail before refiner training.
    first=run/'expert_oof/b4/fold1/probabilities.npz'
    with np.load(first) as archive:bad={n:archive[n] for n in archive.files}
    bad['not_held_out']=p;np.savez_compressed(first,**bad)
    try:prepare(run,data,ROOT/'expert_splits',work/'bad_features')
    except ValueError:pass
    else:raise AssertionError('Bad OOF inventory was accepted.')
write_json(ROOT/'checks/features_cpu.json',dict(ok=True,synthetic_only=True,training_started=False,
    feature_count=70,crops=1120,parameters=11876,optimizer_updates=0,synthetic_cpu_forward=1,
    oof_inventory_guard=True,gray_normalization=True,zero_refiner_identity=True))
print('PASS: synthetic OOF70 features, train-only manifest, crops, grayscale normalization and zero-refiner identity.')
