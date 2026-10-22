"""One isolated attempt, on one assigned GPU (or CPU). No automatic training entry."""
import argparse, copy, os, subprocess, sys, time
from pathlib import Path
from common import (ROOT, BASE, EXPECTED_SCORE, HR_SLOTS, read_json, write_json, environment,
                    weight_key, baseline_weights, records, artifacts, new_attempt)

def set_option(argv,flag,value):
    if flag in argv:argv[argv.index(flag)+1]=str(value)
    else:argv.extend([flag,str(value)])

def command_for(task,destination,parent=None,features=None):
    template=next(t for t in read_json(ROOT/'original_recipe.json')['tasks'] if t['id']==task['recipe'])
    argv=[v.format(source=str(ROOT/'source'),run=str(BASE),workers='4') for v in template['argv']]
    set_option(argv,'--seed',task['seed'])
    set_option(argv,'--out' if '--out' in argv else '--output-dir',destination)
    if parent:set_option(argv,'--init-checkpoint',parent)
    if features:set_option(argv,'--features',features)
    return [sys.executable,'-B','-u',*argv]

def call(command,attempt,name):
    write_json(attempt/(name+'_command.json'),command)
    print('COMMAND:',' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),check=True,cwd=attempt,env=environment(ROOT))

def expert_view(destination,weights):
    destination.mkdir()
    for key,baseline in baseline_weights().items():
        if not key.startswith('E/'):continue
        source=Path(weights.get(key,baseline));target=destination/'expert'/key.removeprefix('E/')
        target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source)
        if source.name=='last.pt':(target.parent/'args.json').symlink_to(source.parent/'args.json')
    return destination

def expert_predict(view,output,family,attempt,fold=None):
    argv=[sys.executable,'-B','-u',ROOT/'expert_worker.py','--bundle',ROOT/'fusion_source','--run',view,
          '--data-root',ROOT/'data','--splits',ROOT/'expert_splits','--family',family,'--output',output]
    if fold:argv += ['--fold',fold]
    elif family=='b4':argv += ['--b4-dir','model329_ws','--fold-ids','1,2,3,4']
    else:argv += ['--fold-ids','2,4' if family=='hr1' else '1,4']
    call(argv,attempt,f'{family}_{fold or "valid"}')

def execute(run,task):
    from engine import baseline_context, evaluate, predict_R, select_re, apply_candidate
    attempt=new_attempt(run,task['id']);started=time.monotonic();completed=records(run)
    record=dict(**task,attempt=str(attempt));paths=[]
    group=task['group'];context=baseline_context()
    if group=='baseline':
        result=evaluate(run,context)
        assert abs(result['image_miou']-EXPECTED_SCORE)<1e-12,result['image_miou']
        from PIL import Image
        import numpy as np
        for row in result['images']:
            name=row['image']
            assert np.array_equal(np.asarray(Image.open(Path(result['evaluation'])/f'predicted_labels/{name}.png')),
                                  np.asarray(Image.open(BASE/f'stage2/predicted_labels/{name}.png'))),name
        record.update(result);paths.append(Path(result['evaluation']))
    elif group=='parents':
        parents=sorted([completed[t] for t in task['deps']],key=lambda r:(-r['image_miou'],r['id']))[:2]
        record['parents']=[r['id'] for r in parents]
    elif group=='select_RE':
        record.update(select_re(run));paths.append(Path(record['evaluation']))
    elif group=='features':
        context=completed['select_RE']['context'];view=expert_view(attempt/'view',context['weights'])
        changed=context['choices'].get('E_hr')
        for family in ['b4','hr1','hr2']:
            for fold in range(1,5):
                target=view/f'expert_oof/{family}/fold{fold}'
                target.parent.mkdir(parents=True,exist_ok=True)
                if changed and family!='b4' and (int(family[-1]),fold) in HR_SLOTS:
                    expert_predict(view,target,family,attempt,fold)
                else:target.symlink_to(BASE/f'expert_oof/{family}/fold{fold}',target_is_directory=True)
        from features import prepare
        features=attempt/'features/train';prepare(view,ROOT/'data',ROOT/'expert_splits',features)
        record.update(features=str(features),context=context,changed_hr=changed,
                      b4_oof='fixed original pre-warmstart B4; never B4ws')
        paths += [features,view/'expert_oof']
    elif group=='select_final':
        candidates=[completed['select_RE'],completed['baseline']]+[completed[t] for t in task['deps']]
        best=max(candidates,key=lambda r:r['image_miou'])
        # Refiner was fit to selected E_hr OOF; keep that HR set during the final recheck.
        if best.get('group')=='F':
            refined=select_re(run,initial=best['context'],refiner_candidate=best,rounds=1)
            if refined['image_miou']>best['image_miou']:best=refined
        final=evaluate(run,best['context'],panels=True)
        assert final['image_miou']>=EXPECTED_SCORE-1e-12
        assert final['image_miou']>=max(r['image_miou'] for r in candidates)-1e-12
        record.update(final);paths.append(Path(final['evaluation']))
    elif group=='export':
        from finalize import export
        record.update(export(run,attempt,completed['select_final']))
        paths.append(Path(record['result']))
    else:
        if task['kind']=='train':
            parent=None
            if group=='347' and task['parent_rank'] is not None:
                parent_record=completed[completed['parents']['parents'][task['parent_rank']]]
                parent=parent_record['checkpoint'];context=copy.deepcopy(parent_record['context'])
                record['parent_task']=parent_record['id']
            features=completed['features']['features'] if group=='F' else None
            train=attempt/'training';argv=command_for(task,train,parent,features)
            call(argv,attempt,'train')
            filename='epoch8.pt' if group=='F' else 'best.pt' if group in ['269','restart51','restart61'] else 'last.pt'
            checkpoint=train/filename
            if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
            key=weight_key(group,task.get('fold'),task.get('slot'))
            context['weights'][key]=str(checkpoint)
            record.update(checkpoint=str(checkpoint),training_dir=str(train),weight_key=key)
            paths.append(checkpoint)
            for name in ['args.json','bn_freeze_check.json','completion.json','recipe.json']:
                if (train/name).exists():paths.append(train/name)
            if group=='E_b4':
                receipt=read_json(train/'bn_freeze_check.json')
                assert receipt['ok'] and receipt['bn_running_stats_frozen']
            if 'fold' in task:
                record['context']=context
                record['artifacts']=artifacts(paths);record['elapsed_seconds']=time.monotonic()-started
                return record
            if group=='F':
                own_weights=context['weights'];context=copy.deepcopy(completed['select_RE']['context'])
                context['weights'].update(own_weights);context['choices']['F']=task['id']
                record['features']=features
        else:
            for identifier in task['deps']:context['weights'].update(completed[identifier]['context']['weights'])
        if group in ['E_b4','E_hr']:
            view=expert_view(attempt/'expert_view',context['weights'])
            for family in (['b4'] if group=='E_b4' else ['hr1','hr2']):
                out=attempt/f'valid_{family}';expert_predict(view,out,family,attempt)
                context['probabilities']['e_'+family]=str(out/'probabilities.npz');paths.append(out)
        elif group!='F':
            component='base' if group.startswith('restart') else group
            prob=predict_R(component,attempt/'inference',context['weights'],group if group.startswith('restart') else None)
            context['probabilities'][group]=str(prob);paths.append(prob)
        from engine import group_name
        context['choices'][group_name(group)]=task['id']
        result=evaluate(run,context)
        record.update(result,candidate=True);paths.append(Path(result['evaluation']))
    record['artifacts']=artifacts(paths);record['elapsed_seconds']=time.monotonic()-started
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--task',required=True);a=p.parse_args()
    if os.environ.get('LITE20_SEED_USER_STARTED')!='1':raise RuntimeError('Use run.sh train --execute')
    from common import require_komap
    require_komap();run=a.run.resolve();task=next(t for t in read_json(run/'plan.json')['tasks'] if t['id']==a.task)
    if task['kind']!='cpu':
        import torch
        assert torch.cuda.device_count()==1
        assert os.environ.get('KOMAP_PHYSICAL_GPU') in ['4','5','6','7']
    output=run/'completed'/f'{task["id"]}.json'
    if output.exists():raise FileExistsError(output)
    result=execute(run,task);write_json(output,result)
    print(f'COMPLETE {task["id"]} image_miou={result.get("image_miou")}',flush=True)

if __name__=='__main__':main()
