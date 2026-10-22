"""Read-only analysis of existing masks and training records. No training/inference."""
from pathlib import Path
import csv, hashlib, json, re, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[1]
RESULT=ROOT/'results/model_results_20261009_2'
EXP=ROOT/'experiments/model349_fusion805_retrain_gpu234567_v1'
RUN=Path(json.loads((RESULT/'complete.json').read_text())['source_run'])
OUT=PACKAGE/'evidence'
CLASSES=['Al3Ni','Primary Si','Al','Eutectic Si']
PALETTE=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8)

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def decode(p):
    a=np.array(Image.open(p).convert('RGB'));v=np.full(a.shape[:2],255,np.uint8)
    for c,color in enumerate(PALETTE):v[(a==color).all(-1)]=c
    assert (v<4).all(),p
    return v
def iou(pred,gt):
    vals=[]
    for c in range(4):
        union=np.count_nonzero((pred==c)|(gt==c))
        vals.append(np.count_nonzero((pred==c)&(gt==c))/union if union else 0.)
    return vals

def main():
    preserved={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'모델링결과서.doc',*RESULT.glob('*.json'),RESULT/f'{RESULT.name}.md',*RESULT.glob('predicted_labels/*.png'),*RESULT.glob('stage2/predicted_labels/*.png')]}
    write(OUT/'protected_input_hashes.json',preserved)
    records=[];histograms={}
    for split in ['train','valid','test']:
        data=ROOT/'dataset' if split=='test' else EXP/'data'
        paths=sorted((data/split/'images').glob('*_image.png'))
        assert len(paths)==dict(train=70,valid=20,test=10)[split]
        histograms[split]=[]
        for p in tqdm(paths,desc=f'Report EDA: {split}',unit='image'):
            gray=np.array(Image.open(p).convert('L'));h=np.bincount(gray.ravel(),minlength=256)
            histograms[split].append(h/h.sum())
            row=dict(split=split,name=p.stem.removesuffix('_image'),specimen=p.name.split('_')[0],path=str(p),sha256=sha(p),
                     width=gray.shape[1],height=gray.shape[0],pixels=gray.size,mean=float(gray.mean()),std=float(gray.std()))
            if split!='test':
                mask=p.parent.parent/'masks'/p.name.replace('_image','_mask')
                gt=decode(mask);assert gt.shape==gray.shape
                count=np.bincount(gt.ravel(),minlength=4)
                row.update(mask=str(mask),mask_sha256=sha(mask),class_pixels=count.tolist(),class_fraction=(count/gt.size).tolist())
            records.append(row)
    summary={}
    for split in ['train','valid','test']:
        rows=[r for r in records if r['split']==split]
        summary[split]=dict(images=len(rows),specimens=sorted({r['specimen'] for r in rows}),pixels=sum(r['pixels'] for r in rows),
            mean=float(np.mean([r['mean'] for r in rows])),std=float(np.mean([r['std'] for r in rows])))
        if split!='test':
            counts=np.array([r['class_pixels'] for r in rows]);frac=np.array([r['class_fraction'] for r in rows])
            summary[split].update(pixel_share=(counts.sum(0)/counts.sum()).tolist(),image_share=frac.mean(0).tolist(),
                present=(counts>0).sum(0).tolist(),absent={CLASSES[c]:[r['name'] for r in rows if r['class_pixels'][c]==0] for c in range(4)})
    overlaps={f'{a}/{b}':sorted(set(summary[a]['specimens'])&set(summary[b]['specimens'])) for a,b in [('train','valid'),('train','test'),('valid','test')]}
    eda=dict(records=records,summary=summary,overlaps=overlaps,histograms={k:np.mean(v,axis=0).tolist() for k,v in histograms.items()})
    write(OUT/'eda.json',eda)
    old=read(ROOT/'experiments/model349_report_materials_v1/outputs/analysis_20261007_1/eda.json')
    assert {r['sha256'] for r in records}=={r['sha256'] for r in old['records']}
    stages={'R':RUN/'release_inference','stage1':RESULT,'stage2':RESULT/'stage2'}
    stored={k:read(p/'metrics.json') for k,p in stages.items()}
    metrics={k:[] for k in ['R','E','stage1','stage2']};changes=[]
    for row in tqdm([r for r in records if r['split']=='valid'],desc='Independent score audit',unit='image'):
        name=row['name'];gt=decode(row['mask']);preds={k:np.array(Image.open(p/f'predicted_labels/{name}.png')) for k,p in stages.items()}
        e=np.load(RUN/f'expert_probabilities/{name}.npy',allow_pickle=False)
        order=np.array([2,0,1,3],np.uint8);preds['E']=order[e[order].argmax(0)]
        for k,pred in preds.items():
            assert pred.shape==gt.shape and set(np.unique(pred))<=set(range(4))
            vals=iou(pred,gt);score=float(np.mean(vals))
            if k in stored:
                ref=next(r for r in stored[k]['images'] if r.get('image',r.get('name'))==name)
                assert np.allclose(vals,ref['class_iou'],atol=1e-12,rtol=0)
                assert abs(score-ref['image_miou'])<1e-12
            metrics[k].append(dict(image=name,class_iou=vals,image_miou=score,mask_sha256=sha(stages[k]/f'predicted_labels/{name}.png') if k in stages else None))
        for before,after in [('R','stage1'),('stage1','stage2')]:
            a,b=preds[before],preds[after];changed=a!=b
            changes.append(dict(image=name,transition=f'{before}→{after}',pixels=gt.size,changed=int(changed.sum()),
                corrected=int(((a!=gt)&(b==gt)).sum()),worsened=int(((a==gt)&(b!=gt)).sum()),
                wrong_to_wrong=int((changed&(a!=gt)&(b!=gt)).sum()),
                delta_pp=(np.mean(iou(b,gt))-np.mean(iou(a,gt)))*100))
    scored={}
    for k,rows in metrics.items():
        scored[k]=dict(image_miou=float(np.mean([r['image_miou'] for r in rows])),class_iou=np.mean([r['class_iou'] for r in rows],axis=0).tolist(),images=rows)
        if k in stored:assert abs(scored[k]['image_miou']-stored[k]['image_miou'])<1e-12
    change_summary={}
    for trans in ['R→stage1','stage1→stage2']:
        rows=[r for r in changes if r['transition']==trans]
        change_summary[trans]={k:sum(r[k] for r in rows) for k in ['pixels','changed','corrected','worsened','wrong_to_wrong']}
        change_summary[trans].update(improved=sum(r['delta_pp']>1e-10 for r in rows),declined=sum(r['delta_pp']<-1e-10 for r in rows),
            unchanged=sum(abs(r['delta_pp'])<=1e-10 for r in rows),worst=min(rows,key=lambda r:r['delta_pp']),best=max(rows,key=lambda r:r['delta_pp']))
    write(OUT/'audited_metrics.json',scored);write(OUT/'pixel_changes.json',dict(summary=change_summary,images=changes))
    plan=read(RESULT/'plan.json');events=[json.loads(l) for l in (RESULT/'events.jsonl').read_text().splitlines()]
    tasks=[]
    for task in plan['tasks']:
        start=next(e for e in events if e['task']==task['id'] and e['event']=='started')
        end=next(e for e in events if e['task']==task['id'] and e['event']=='finished')
        tasks.append(dict(id=task['id'],role=task['role'],family=task.get('family'),gpu=start['gpu'],seed=task.get('seed'),start=start['elapsed_seconds'],end=end['elapsed_seconds'],seconds=end['elapsed_seconds']-start['elapsed_seconds'],argv=task['argv']))
    status=read(RUN/'status.json');finish=read(RUN/'finished.json')
    timing=dict(tasks=tasks,pipeline_seconds=status['elapsed_seconds'],finalization_seconds=finish['finalization_seconds'],
        total_seconds=status['elapsed_seconds']+finish['finalization_seconds'],gpu_job_seconds=sum(t['seconds'] for t in tasks if t['gpu'] is not None),
        refiner=read(RESULT/'refiner_training_receipt.json'))
    write(OUT/'runtime.json',timing)
    for name in ['metrics.json','fusion_comparison.json','complete.json','provenance.json','preflight.json','plan.json','refiner_training_receipt.json','environment.freeze.txt']:
        shutil.copy2(RESULT/name,OUT/('local_'+name))
    shutil.copy2(RESULT/f'{RESULT.name}.md',OUT/'local_result_report.md')
    shutil.copy2(RUN/'expert/refiner/progress.json',OUT/'refiner_loss.json')
    shutil.copy2(RUN/'refiner_features/train/manifest.json',OUT/'refiner_oof_manifest.json')
    shutil.copy2(RESULT/'selected_model/model_manifest.json',OUT/'trained_model_manifest.json')
    for stage,source in [('stage1',RESULT),('stage2',RESULT/'stage2')]:
        dest=PACKAGE/'appendix'/stage;dest.mkdir(exist_ok=True)
        for p in sorted((source/'valid_mask_comparison').glob('*.png')):shutil.copy2(p,dest/p.name)
    inventory=[]
    for p in sorted((ROOT/'results').glob('model_results_*/*.md')):
        if p.stem!=p.parent.name:continue
        txt=p.read_text(errors='replace');inventory.append(dict(result=p.parent.name,title=txt.splitlines()[0].lstrip('# '),path=str(p.relative_to(ROOT))))
    write(OUT/'project_result_inventory.json',inventory)
    with (PACKAGE/'tables/project_result_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['result','title','path']);w.writeheader();w.writerows(inventory)
    write(PACKAGE/'validation/numeric_audit.json',dict(ok=True,verified_images=20,verified_stored_stages=3,verified_class_ious=240,
        expert_score_from_saved_probabilities=True,old_eda_image_hashes_match=True,training_started=False,new_gpu_inference=False,
        global_miou_computed=False,figure_panels_copied=40))
    print(json.dumps(dict(scores={k:v['image_miou']*100 for k,v in scored.items()},overlaps=overlaps,changes=change_summary,total_hours=timing['total_seconds']/3600),ensure_ascii=False,indent=2,default=lambda x:x.item()))

if __name__=='__main__':main()
