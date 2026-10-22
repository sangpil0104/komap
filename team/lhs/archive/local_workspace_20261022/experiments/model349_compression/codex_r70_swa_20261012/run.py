import os, sys, json, time, hashlib, subprocess, shutil, traceback
from pathlib import Path
import yaml
K=Path.home()/'komap'; O=K/'codex_r70_swa_20261012'; O.mkdir(exist_ok=True)
PY=sys.executable; B=K/'Model349_lite20_b4ws_bundle'; PKG=B/'Model349_fusion805'
SCRIPT=O/'train_swa.py'; START=time.time()
def dump(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def run(cmd,log,cwd=None,env=None):
    dump(O/'active_child.json',{'stage':str(log),'command':list(map(str,cmd)),'started':time.time()})
    with log.open('a') as f:
        p=subprocess.Popen(list(map(str,cmd)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        dump(O/'active_child.json',{'stage':str(log),'pid':p.pid,'command':list(map(str,cmd)),'started':time.time()})
        if p.wait()!=0:raise RuntimeError('failed: '+str(log))
def main():
    if (O/'completion.json').exists():return
    # Already-running restart61 trainer keeps its original loaded code. Evaluate saved epochs here, without retraining.
    select=K/'codex_restart61_globalstats_20261012'
    if (select/'training_complete.json').exists() and not (select/'epoch_selection_completion.json').exists():
        for f in ('result.json','completion.json'):
            if (select/f).exists() and not (select/('epoch10_only_'+f)).exists():shutil.copy2(select/f,select/('epoch10_only_'+f))
        with (select/'epoch_selection.log').open('a') as log:
            proc=subprocess.run([PY,'-u',select/'run.py','--select-epochs'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:print('RESTART61_EPOCH_SELECTION_FAILED: see epoch_selection.log; proceeding with separately authorized SWA',flush=True)
    # Queue guard: do not overlap any existing compute process on this single GPU.
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        print('waiting for compute GPU to be idle',flush=True);time.sleep(20)
    initial={}
    for m in ('322','326','329'):
        src=K/'r70/full70'/f'Model349_lite20_r70_B4_{m}'
        for f in ('last.pt','args.json'):initial[str(src/f)]=sha(src/f)
    initial[str(SCRIPT)]=sha(SCRIPT);dump(O/'input_sha.json',initial)
    common=B/'train_route/experiments/model326_hybrid_fullstats/KoMaP'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',NO_ALBUMENTATIONS_UPDATE='1',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(common)+':'+str(PKG/'Model349_exh14/oof_design/runtime')
    for m in ('322','326','329'):
        src=K/'r70/full70'/f'Model349_lite20_r70_B4_{m}'; out=O/'weights'/m;out.mkdir(parents=True,exist_ok=True)
        if (out/'training_complete.json').exists():continue
        if (out/'last.pt').exists():raise RuntimeError('Unverified partial output exists: '+str(out))
        args=json.loads((src/'args.json').read_text())
        cmd=[PY,'-u',SCRIPT,'--data',K/'data','--out',out,'--train_splits','train','--init_checkpoint',src/'last.pt']
        for name in ('arch','encoder','norm','input','crop','scale','target_width','sample','bs','wd','dice_w','cldice_w','seed'):
            cmd+=['--'+name,str(args[name])]
        cmd+=['--swa_iters','4000','--swa_lr','1e-4','--swa_every','500','--bn_batches','300','--workers','4','--amp_init_scale','1024','--no_valid_eval','--deterministic_workers']
        run(cmd,out/'train.log',cwd=common,env=env)
        assert 'SWA done: 8 snapshots' in (out/'train.log').read_text()
        dump(out/'training_complete.json',{'sha256':sha(out/'last.pt'),'finished':time.time()})
    # Isolated inference package; published paths and release checkpoints remain untouched.
    parent=O/'parent'
    if not parent.exists():shutil.copytree(PKG,parent,ignore=shutil.ignore_patterns('weights','__pycache__'))
    exh=parent/'Model349_exh14'; aliases={'322':'model322_team_b4_group_oof','326':'model326_hybrid_fullstats_group_oof','329':'model329_fullstats_width4000_uniform_group_oof'}
    orig=K/'l18h/e2e_valid/release'; expert=K/'l18h/e2e_valid/expert'
    assert (K/'l18h/e2e_valid/completion.json').exists()
    assert expert.exists()
    metrics={'lite18h':json.loads((K/'l18h/e2e_valid_metrics.json').read_text())}
    for variant in ('train70','swa'):
        wr=O/('weights_'+variant);wr.mkdir(exist_ok=True)
        for m,dirname in aliases.items():
            link=wr/(dirname+'_full70')
            target=O/'weights'/m if variant=='swa' else K/'r70/full70'/f'Model349_lite20_r70_B4_{m}'
            if not link.exists():link.symlink_to(target,target_is_directory=True)
        wl=exh/'weights'
        if wl.is_symlink():wl.unlink()
        elif wl.exists():raise RuntimeError('unexpected inference weight directory')
        wl.symlink_to(wr,target_is_directory=True)
        work=O/variant/'R_valid';work.mkdir(parents=True,exist_ok=True);(work/'logs').mkdir(exist_ok=True);(work/'probabilities').mkdir(exist_ok=True)
        manifest=json.loads((orig/'input_manifest.json').read_text())
        cfg=yaml.safe_load((B/'Model349_fusion805_lite23_b4ws/selected_R_lite18h.yaml').read_text());cfg['inference']['b4_full70']=['322','326','329']
        manifest['config']=cfg;dump(work/'input_manifest.json',manifest)
        for c in ('base','347','269'):
            link=work/'probabilities'/c
            if not link.exists():link.symlink_to(orig/'probabilities'/c,target_is_directory=True)
        for c in ('322','326','329'):
            if not (work/f'{c}_complete.json').exists():
                run([PY,'-B','-u',exh/'infer_lite.py','--_worker',c,'--output',work],work/'logs'/f'{c}.log',cwd=exh,env=env)
                dump(work/f'{c}_complete.json',{'finished':time.time()})
        if not (work/'assemble_complete.json').exists():
            code="import infer_lite as m;from pathlib import Path;m.assemble(Path(__import__('sys').argv[1]),Path(__import__('sys').argv[2]))"
            run([PY,'-B','-c',code,work,K/'data/valid/masks'],work/'logs/assemble.log',cwd=exh,env=env)
            dump(work/'assemble_complete.json',{'finished':time.time()})
        fusion=O/variant/'fusion_valid'
        if not (fusion/'completion.json').exists():
            run([PY,'-B',parent/'infer_fusion.py','--input',K/'data/valid/images','--release-labels',work/'predicted_labels','--expert-probabilities',expert,'--refiner',K/'weights/all/assets/refiner/epoch8.pt','--output',fusion,'--device','cpu'],O/variant/'fusion.log',env=env)
        mp=O/variant/'valid_metrics.json'
        run(['bash',B/'score.sh',fusion/'predicted_labels',K/'data/valid/masks',mp],O/variant/'score.log',env=dict(env,PYTHON=PY))
        metrics[variant]=json.loads(mp.read_text())
    for p,s in initial.items():assert sha(Path(p))==s,'Input changed: '+p
    result={'scores_percent':{k:v['image_miou']*100 for k,v in metrics.items()},'elapsed_seconds':time.time()-START,'fixed_R_E_F_settings':True,'valid_used_for_training':False,'release_replaced':False,'input_sha_unchanged':True}
    for reference in ('train70','lite18h'):
        a={x['image']:x['image_miou'] for x in metrics['swa']['images']};b={x['image']:x['image_miou'] for x in metrics[reference]['images']}
        delta={k:100*(a[k]-b[k]) for k in a};result['swa_vs_'+reference]={'mean_delta_pp':sum(delta.values())/len(delta),'per_image_delta_pp':delta,'improved':sum(x>0 for x in delta.values()),'worse':sum(x<0 for x in delta.values()),'worst_image':min(delta,key=delta.get)}
    dump(O/'result.json',result);dump(O/'completion.json',{'finished':time.time(),'result':'result.json','input_sha_unchanged':True});print(json.dumps(result['scores_percent']),flush=True)
try:main()
except Exception:
    dump(O/'failure.json',{'time':time.time(),'error':traceback.format_exc()});raise
