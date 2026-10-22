"""Independent mask scoring, paired losses, storage and repeated timings."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def load(p): return json.loads(p.read_text())
def dump(p, obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def score(run, gt):
    manifest=load(run/'input_manifest.json'); config=manifest['config']
    palette=np.asarray(config['palette'],dtype=np.uint8)
    rows=[]
    for r in manifest['images']:
        name=r['name']; pred=np.asarray(Image.open(run/'predicted_labels'/f'{name}.png'))
        paths=[gt/f'{name}_mask.png',gt/f'{name}.png']; target=next(p for p in paths if p.is_file())
        raw=np.asarray(Image.open(target))
        if raw.ndim==2 and np.isin(raw,range(4)).all(): labels=raw
        else:
            rgb=np.asarray(Image.open(target).convert('RGB'))
            labels=np.full(rgb.shape[:2],255,dtype=np.uint8)
            for i,color in enumerate(palette): labels[np.all(rgb==color,axis=-1)]=i
        assert not np.any(labels==255) and pred.shape==labels.shape and np.isin(pred,range(4)).all()
        ious=[]
        for c in range(4):
            a=pred==c; b=labels==c
            union=int(np.count_nonzero(a|b)); intersection=int(np.count_nonzero(a&b))
            ious.append(intersection/union if union else 0.0)
        rows.append({'image':name,'class_iou':ious,'image_miou':float(np.mean(ious))})
    assert len(rows)==20 and len({r['image'] for r in rows})==20
    metrics=load(run/'metrics.json'); vals=np.asarray([r['class_iou'] for r in rows]); old={r['image']:r for r in metrics['images']}
    assert set(old)=={r['image'] for r in rows}
    assert np.allclose([old[r['image']]['class_iou'] for r in rows], vals, rtol=0,atol=1e-12)
    assert abs(vals.mean()-metrics['image_miou'])<1e-12
    return {'miou':float(vals.mean()),'classes':vals.mean(axis=0).tolist(),'images':{r['image']:r for r in rows},'input_sha256':{r['name']:r['sha256'] for r in manifest['images']},'calibration_sha256':manifest['calibration_sha256']}
def pair(refpath, path, a,b):
    assert a['input_sha256']==b['input_sha256'] and a['calibration_sha256']==b['calibration_sha256']
    deltas={n:100*(b['images'][n]['image_miou']-a['images'][n]['image_miou']) for n in a['images']}
    class_deltas={n:(100*(np.asarray(b['images'][n]['class_iou'])-a['images'][n]['class_iou'])).tolist() for n in a['images']}
    changed=total=0
    for n in a['images']:
        x=np.asarray(Image.open(refpath/'predicted_labels'/f'{n}.png')); y=np.asarray(Image.open(path/'predicted_labels'/f'{n}.png'))
        changed+=int(np.count_nonzero(x!=y)); total+=x.size
    mean=100*(b['miou']-a['miou']); worst=min(deltas,key=deltas.get)
    return {'delta_pp':mean,'class_delta_pp':(100*(np.asarray(b['classes'])-a['classes'])).tolist(),'worst_image':worst,'worst_image_delta_pp':deltas[worst],'per_image_delta_pp':deltas,'per_image_class_delta_pp':class_deltas,'mask_changed_pixels':changed,'mask_total_pixels':total,'mask_agreement':1-changed/total,'quality_pass':mean>=-0.01-1e-10 and min(deltas.values())>=-0.10-1e-10}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--gt',type=Path,required=True); args=p.parse_args()
    root=args.root; refpath=root/'runs/lite15'; a=score(refpath,args.gt)
    historical={name:(root/'runs'/name,score(root/'runs'/name,args.gt)) for name in ('historical349','historical14')}
    report={'metric':'mean of 20 per-image means of 4 IoUs; empty union=0','class_order':load(refpath/'input_manifest.json')['config']['class_order'],'reference':'lite15','runs':{}}
    for path in sorted((root/'runs').iterdir()):
        if not (path/'completion.json').is_file() or not (path/'metrics.json').is_file(): continue
        b=score(path,args.gt); completion=load(path/'completion.json'); key=path.name; bundle='lite15' if key.startswith(('batched_control','hramp_control')) else key.split('_repeat')[0]
        manifest=root/'bundles'/bundle/'checkpoint_manifest.json'
        item={'miou_percent':100*b['miou'],'class_iou_percent':(100*np.asarray(b['classes'])).tolist(),'completion':completion,'vs_same_gpu_lite15':pair(refpath,path,a,b)}
        for name,(oldpath,oldscore) in historical.items(): item['vs_'+name]=pair(oldpath,path,oldscore,b)
        if manifest.is_file(): item['checkpoints']=load(manifest)
        report['runs'][key]=item
    dump(root/'comparison.json',report)
    print(json.dumps({k:{'miou':v['miou_percent'],'seconds':v['completion']['elapsed_seconds'],'delta_pp':v['vs_same_gpu_lite15']['delta_pp'],'worst_delta_pp':v['vs_same_gpu_lite15']['worst_image_delta_pp'],'quality_pass':v['vs_same_gpu_lite15']['quality_pass']} for k,v in report['runs'].items()},indent=2))
if __name__=='__main__': main()
