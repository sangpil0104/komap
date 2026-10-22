"""Read-only checks; never initializes a GPU model or starts training."""
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from common import ROOT,MODEL,BASE,read_json,B4_DIRS,command_for

PALETTE=np.asarray([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8)

def decode(path):
    with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
    y=np.full(rgb.shape[:2],255,np.uint8)
    for c,color in enumerate(PALETTE):y[np.all(rgb==color,axis=-1)]=c
    if np.any(y==255):raise ValueError(f'Unknown GT color: {path}')
    return y

def dataset_check(root=ROOT):
    counts={};names={}
    for split,n in [('train',70),('valid',20),('test',10)]:
        images=sorted((root/f'data/{split}/images').glob('*_image.png'))
        assert len(images)==n,(split,len(images))
        names[split]={p.stem.removesuffix('_image') for p in images};counts[split]=len(images)
        if split=='test':continue
        masks=list((root/f'data/{split}/masks').glob('*_mask.png'));assert len(masks)==n
        for p in tqdm(images,desc=f'Check {split}',unit='image',mininterval=5):
            target=decode(root/f'data/{split}/masks/{p.stem.removesuffix("_image")}_mask.png')
            with Image.open(p) as im:assert target.shape==(im.height,im.width)
    assert not (names['train']&names['valid'] or names['train']&names['test'] or names['valid']&names['test'])
    assert set(p.name for p in (root/'train_data').iterdir())=={'train'}
    assert (root/'train_data/train').resolve()==(root/'data/train').resolve()
    return dict(counts=counts,train_only_data_view=True,train_valid_test_disjoint=True)

def recipe_check(root=ROOT):
    compact=root/'compact_source'
    configs={v:yaml.safe_load((compact/f'selected_R_{v}.yaml').read_text()) for v in ['lite20','lite20_r70']}
    assert configs['lite20']['route']==configs['lite20_r70']['route']
    a=dict(configs['lite20']['inference']);b=dict(configs['lite20_r70']['inference'])
    assert a.pop('b4_fold_ids')=={'322':[3],'326':[1,2],'329':[1,3,4]}
    assert b.pop('b4_full70')==['322','326','329'] and a==b
    for key in ['route','inference','palette','calibration']:
        assert configs['lite20'][key]==yaml.safe_load((MODEL/'lite20/selected_R_lite20.yaml').read_text())[key]
    e=[read_json(compact/f'selected_config_{v}.json')['expert'] for v in ['lite20','lite20_r70']]
    assert e[0]==e[1]==read_json(MODEL/'lite20/selected_config_lite20.json')['expert']
    for c in B4_DIRS:
        cmd=command_for(c,Path('/not-started'))
        assert cmd[cmd.index('--iters')+1]=='21000' and cmd[cmd.index('--train_splits')+1]=='train'
        assert '--no_valid_eval' in cmd and '--init_checkpoint' not in cmd and '--deterministic_workers' in cmd
        assert cmd[cmd.index('--seed')+1]=='0'
    baseline=read_json(BASE/'stage2/metrics.json')['image_miou']
    assert abs(baseline-.8025584552762022)<1e-12
    return dict(training_count=3,iterations=21000,seed=0,r70_weights=17,control_weights=20,
                only_R_B4_changes=True,baseline_REF_image_miou=baseline,baseline_R_image_miou=.8001013505955459)
