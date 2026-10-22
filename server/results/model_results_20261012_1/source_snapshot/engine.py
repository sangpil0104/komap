"""Exact frozen R routing, family FP16 quantization and two-stage R/E/F scoring."""
import copy, fcntl, pickle, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from common import ROOT, BASE, MODEL, read_json, write_json, signature, baseline_weights, sha
from features import blend, decode, PALETTE
from report_utils import class_ious, comparison

sys.path.insert(0,str(ROOT/'frozen'))
import infer_lite as frozen

def samples():
    rows=copy.deepcopy(read_json(BASE/'release_inference/input_manifest.json')['images'])
    for row in rows:
        row['path']=str(ROOT/f'data/valid/images/{row["name"]}_image.png')
        row['gt']=str(ROOT/f'data/valid/masks/{row["name"]}_mask.png')
        with Image.open(row['path']) as im:mean=float(np.asarray(im.convert('RGB'))[...,0].mean())
        row['high_family']='restart61' if 159<=mean<161 else 'restart51'
    return rows

def baseline_context():return dict(probabilities={},weights={},choices={})

def verify_cache(root):
    receipt=read_json(root/'complete.json')
    for name,expected in receipt['files_sha256'].items():
        if sha(root/name)!=expected:raise ValueError(f'Changed cache: {root/name}')

def seal_cache(root,**metadata):
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['complete.json','lock']}
    write_json(root/'complete.json',dict(**metadata,files_sha256=files))

def group_name(group):return 'HR_pair' if group in ['336','347'] else group

def apply_candidate(context,candidate,group):
    result=copy.deepcopy(context);group=group_name(group)
    components={'HR_pair':['336','347'],'E_b4':['e_b4'],'E_hr':['e_hr1','e_hr2'],'F':[]}.get(group,[group])
    from common import weight_key
    keys=([weight_key('336'),weight_key('347')] if group=='HR_pair' else
          [k for k in baseline_weights() if (group=='E_b4' and k.startswith('E/model329_ws/')) or
           (group=='E_hr' and k.startswith('E/model392/'))])
    if group not in ['HR_pair','E_b4','E_hr']:
        keys=[weight_key(group,f) for f in ([1,2] if group=='326' else [3] if group=='322' else [None])]
    for k in components:result['probabilities'].pop(k,None)
    for k in keys:result['weights'].pop(k,None)
    result['choices'].pop(group,None)
    if candidate:
        result['probabilities'].update(candidate['context']['probabilities'])
        result['weights'].update(candidate['context']['weights'])
        result['choices'][group]=candidate['id']
    return result

def predict_R(component,destination,replacements, family=None):
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(2)
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    destination.mkdir();view=destination/'weights';view.mkdir()
    for key,baseline in baseline_weights().items():
        if not key.startswith('R/'):continue
        path=Path(replacements.get(key,baseline));target=destination/key.removeprefix('R/')
        target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(path)
        if '/fold' in key:(target.parent/'args.json').symlink_to(path.parent/'args.json')
    config=copy.deepcopy(frozen.load_config(ROOT/'frozen/configs/model349_lite.yaml'))
    config['weights_dir']=str(view)
    rows=[r for r in samples() if family is None or r['high_family']==family]
    assert rows
    write_json(destination/'input_manifest.json',dict(config=config,images=rows))
    (destination/'probabilities').mkdir();frozen.worker(destination,component)
    return destination/'probabilities'/component

def expert_fields(run,context):
    """Cache image-only expert/refined hard masks. CPU forward exactly matches infer_fusion."""
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'fusion_source/refiner'))
    from residual_model import Residual
    from gray_input import prepare_gray
    paths={f:context['probabilities'].get('e_'+f,str(BASE/f'expert_valid/{f}/probabilities.npz')) for f in ['b4','hr1','hr2']}
    checkpoint=context['weights'].get('E/refiner/epoch8.pt',str(MODEL/'assets/refiner/epoch8.pt'))
    key=signature(dict(paths=paths,checkpoint=checkpoint));root=run/'field_cache'/key;root.mkdir(parents=True,exist_ok=True)
    with (root/'lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/'complete.json').exists() and 'files_sha256' in read_json(root/'complete.json'):
            verify_cache(root);return root
        model=Residual().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
        arrays={k:np.load(v,allow_pickle=False) for k,v in paths.items()}
        try:
            for row in tqdm(samples(),desc='E/F valid fields',unit='image',mininterval=5):
                name=row['name'];p=blend(*(arrays[f][name] for f in ['b4','hr1','hr2']))
                with Image.open(row['path']) as im:gray=prepare_gray(np.asarray(im.convert('L'),np.float32)/255.)
                with torch.no_grad():value,delta=model(torch.from_numpy(np.log(np.maximum(p,1e-7))[None]),torch.from_numpy(gray[None,None]))
                assert torch.isfinite(value).all() and delta.abs().max()<=1
                def labels(a):return np.asarray([2,0,1,3],np.uint8)[a[[2,0,1,3]].argmax(0)]
                np.savez_compressed(root/f'{name}.npz',expert=labels(p),refined=labels(value.numpy()[0]))
        finally:
            for a in arrays.values():a.close()
        seal_cache(root,paths=paths,checkpoint=checkpoint,images=20,gt_read=False)
    return root

def evaluate(run,context,*,panels=False,force=False):
    import cv2
    cv2.setNumThreads(2)
    out=run/'evaluations'/(signature(context)+('_panels' if panels else ''));out.mkdir(parents=True,exist_ok=True)
    metric_path=out/'metrics.json'
    if metric_path.exists() and (out/'complete.json').exists() and not force and not panels:
        verify_cache(out);return read_json(metric_path)
    fields=expert_fields(run,context);config=frozen.load_config(ROOT/'frozen/configs/model349_lite.yaml')
    (out/'predicted_labels').mkdir(exist_ok=True)
    if panels:(out/'valid_mask_comparison').mkdir(exist_ok=True)
    rows=[];r_changed=any(not k.startswith('e_') for k in context['probabilities'])
    for row in tqdm(samples(),desc='Full R+E+F image mIoU',unit='image',mininterval=5):
        name=row['name']
        if r_changed:
            base=Path(context['probabilities'].get(row['high_family'],BASE/'release_inference/probabilities/base'))/f'{name}.pkl'
            with base.open('rb') as stream:item=pickle.load(stream)
            sources={c:np.load(Path(context['probabilities'].get(c,BASE/f'release_inference/probabilities/{c}'))/f'{name}.npy').astype(np.float32)
                     for c in ['322','326','329','336','347','269']}
            pred=frozen.predict_fixed(item,sources,config).astype(np.uint8)
        else:
            with Image.open(BASE/f'release_inference/predicted_labels/{name}.png') as im:pred=np.asarray(im).copy()
        with np.load(fields/f'{name}.npz') as z:
            expert,refined=z['expert'],z['refined']
            stage1=np.where(np.isin(expert,[0,2]),expert,pred)
            pred=np.where(refined==2,refined,stage1).astype(np.uint8)
        target=decode(row['gt']);ious=class_ious(pred,target);score=float(np.mean(ious))
        Image.fromarray(pred).save(out/f'predicted_labels/{name}.png')
        if panels:
            with Image.open(row['path']) as im:comparison(im,pred,target,PALETTE,name,score).save(out/f'valid_mask_comparison/{name}.png')
        rows.append(dict(image=name,image_miou=score,class_iou=ious))
    assert len(rows)==20
    result=dict(image_miou=float(np.mean([r['image_miou'] for r in rows])),images=rows,image_count=20,
                metric='mean of image-wise four-class IoU; empty union=0',context=context,evaluation=str(out))
    write_json(metric_path,result);seal_cache(out);return result

def candidate_pool(run):
    from common import records
    return [r for r in records(run).values() if r.get('candidate') and r['group']!='F']

def select_re(run,initial=None,refiner_candidate=None,rounds=2):
    pool=candidate_pool(run);base=baseline_context()
    if refiner_candidate:base=apply_candidate(base,refiner_candidate,'F')
    options={}
    for group in ['336','347','269','restart51','restart61','326','322','E_b4','E_hr']:
        if refiner_candidate and group=='E_hr':continue
        best=sorted([r for r in pool if r['group']==group],key=lambda r:(-r['image_miou'],r['id']))[:2]
        options.setdefault(group_name(group),[]).extend(best)
    current=copy.deepcopy(initial or base)
    if refiner_candidate:current=apply_candidate(current,refiner_candidate,'F')
    # Include baseline and every measured single replacement as safe fallback.
    contenders=[evaluate(run,current)] if refiner_candidate else [evaluate(run,current),evaluate(run,base)]
    if not refiner_candidate:contenders += [evaluate(run,r['context']) for r in pool]
    current=max(contenders,key=lambda r:r['image_miou'])['context'];trace=[]
    for iteration in range(rounds):
        changed=False
        for group,choices in options.items():
            before=evaluate(run,current);best=before
            for candidate in [None,*choices]:
                ctx=apply_candidate(current,candidate,group);measured=evaluate(run,ctx)
                if measured['image_miou']>best['image_miou']+1e-12:best=measured
            if best['context']!=current:changed=True
            current=best['context'];trace.append(dict(round=iteration,group=group,score=best['image_miou'],choices=current['choices']))
        if not changed:break
    return dict(**evaluate(run,current),trace=trace)
