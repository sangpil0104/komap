"""Reproduce fixed train-only Model397 OOF metric and export raw log-prob fields."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from fields_io import api,sha,write
ROOT=Path(__file__).resolve().parent;DATA=Path('/workspace/komap_model1')
def main():
 audit=json.loads((ROOT/'oof_input_audit.json').read_text())
 assert audit['images']==70 and audit['groups']==59 and not audit['official_valid_or_test_inputs_opened']
 lock=json.loads((DATA/'experiments/results/model397_fixed_b4_model392_blend_lock.json').read_text())
 assert sha(DATA/'experiments/results/model397_fixed_b4_model392_blend_lock.json')==audit['model397_lock_sha256']
 parent=ROOT.parent/'accuracy81_residual_20261008/runs/teacher_train70'
 manifest=json.loads((parent/'input_manifest.json').read_text());images={x['name']:x for x in manifest['images']}
 palette=np.array(manifest['config']['palette'],dtype=np.uint8);entry=api()[0]
 out=ROOT/'features/train';out.mkdir(parents=True,exist_ok=False);rows=[];scores=[]
 for fold in audit['folds']:
  f=fold['fold'];paths=[DATA/f'outputs/model329_fullstats_width4000_uniform_group_oof/fold{f}_oof/probabilities.npz']
  paths += [DATA/f'outputs/model392_widthnorm/seed{s}/fold{f}_oof/valid_probabilities.npz' for s in [1,2]]
  for p in paths:assert sha(p)==fold['probability_archive_sha256'][str(p.relative_to(DATA))]
  with np.load(paths[0],allow_pickle=False) as b4,np.load(paths[1],allow_pickle=False) as h1,np.load(paths[2],allow_pickle=False) as h2:
   assert set(b4.files)==set(h1.files)==set(h2.files)
   for name in sorted(b4.files):
    im=DATA/'data/train/images'/f'{name}_image.png';gt=DATA/'data/train/masks'/f'{name}_mask.png'
    assert sha(im)==images[name]['sha256'];assert sha(gt)==lock['locked_files'][str(gt.relative_to(DATA))]['sha256']
    gray=np.array(Image.open(im).convert('L'),dtype=np.float32)/255.;target=entry.decode_gt(gt,palette)
    components=[np.asarray(z[name],dtype=np.float32) for z in [b4,h1,h2]]
    for p in components:
     assert p.shape==(4,*gray.shape) and np.isfinite(p).all() and p.min()>=-1e-4 and p.max()<=1.0001 and np.max(np.abs(p.sum(0)-1))<=.03
    # Exactly the fixed original arithmetic blend, no extra normalization.
    probability=(.5*components[0]+.25*components[1]+.25*components[2])[[1,2,0,3]]
    pred=probability.argmax(0);iou=[]
    for c in range(4):
     union=np.count_nonzero((pred==c)|(target==c));inter=np.count_nonzero((pred==c)&(target==c));iou.append(float(inter/union) if union else 0.)
    scores.append({'image':name,'fold':f,'class_iou':iou,'image_miou':float(np.mean(iou))})
    logits=np.log(np.maximum(probability,1e-7)).astype(np.float32)
    p=out/f'{name}.npz';np.savez_compressed(p,logits=logits,gray=gray,target=target)
    rows.append({'name':name,'fold':f,'image_sha256':sha(im),'train_mask_sha256':sha(gt),'features_sha256':sha(p)})
    print('OOF FIELDS',name,flush=True)
 assert len(rows)==70 and len({x['name'] for x in rows})==70 and set(x['name'] for x in rows)==set(audit['all_image_ids'])
 score=float(np.mean([x['image_miou'] for x in scores]));assert abs(100*score-79.9858620080774)<1e-9,(score,'fixed OOF metric differs')
 write(ROOT/'oof_metric_reproduction.json',{'miou_percent':100*score,'class_order':['Al3Ni','PrimarySi','Al','EutecticSi'],'images':scores,'empty_union':0,'input_audit_sha256':sha(ROOT/'oof_input_audit.json'),'official_valid_or_test_opened':False})
 write(out/'manifest.json',{'split':'train','images':sorted(rows,key=lambda x:x['name']),'parent_masks_exact':False,'source_network_components_group_oof':True,'original_fixed_oof_metric_reproduced':True,'raw_oof_miou_percent':100*score,'postprocessing_prior_not_applied':True,'config':manifest['config'],'source_inference_sha256':sha(ROOT.parent/'Model349_exh14/infer_lite.py'),'valid_gt_decoded':False})
 print('OOF METRIC REPRODUCED',100*score,flush=True)
if __name__=='__main__':main()
