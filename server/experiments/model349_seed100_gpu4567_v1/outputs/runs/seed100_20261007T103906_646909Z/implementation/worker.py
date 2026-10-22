"""One isolated training/fusion task, invoked only by the user-started scheduler."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT, BASE, COMPONENT_DIRS, read_json, write_json, environment

def set_option(argv, flag, value):
    if flag in argv: argv[argv.index(flag)+1] = str(value)
    else: argv.extend([flag,str(value)])

def command_for(task, destination, parent_checkpoint=None):
    recipe = read_json(ROOT/'original_recipe.json')
    model=task['model']
    recipe_id=f'train_model{model}' + (f'_fold{task["fold"]}' if 'fold' in task else '')
    template=next(t for t in recipe['tasks'] if t['id']==recipe_id)
    argv=[s.format(source=str(ROOT/'source'),run=str(BASE),workers='4') for s in template['argv']]
    set_option(argv,'--seed',task['seed'])
    set_option(argv,'--out' if 'fold' in task else '--output-dir',destination)
    if parent_checkpoint: set_option(argv,'--init-checkpoint',parent_checkpoint)
    if 'fold' in task: argv.append('--deterministic_workers')
    return [sys.executable,'-B','-u']+argv

def run_command(command, attempt, label):
    write_json(attempt/f'{label}_command.json',command)
    print('COMMAND:', ' '.join(command),flush=True)
    subprocess.run(command,cwd=attempt,env=environment(),check=True)

def execute_task(run, task):
    from engine import evaluate, archive_to_npy, ema_probabilities
    identifier=task['id']; model=task['model']
    if task['kind']=='select':
        records=[read_json(run/'completed'/f'{name}.json') for name in task['deps']]
        ranked=sorted(records,key=lambda r:(-r['image_miou'],r['seed']))
        write_json(run/'parents.json',ranked[:2])
        return dict(id=identifier,kind='select',parent_ids=[r['id'] for r in ranked[:2]])
    parent=None
    if 'parent_rank' in task:
        parent=read_json(run/'parents.json')[task['parent_rank']]
    job=run/'jobs'/identifier; job.mkdir(parents=True,exist_ok=True)
    index=1
    while (job/f'attempt_{index}').exists(): index+=1
    attempt=job/f'attempt_{index}'; attempt.mkdir()
    started=time.monotonic()
    selection={}; replacements={}
    if task['kind']=='train':
        train=attempt/'training'
        command=command_for(task,train, parent['checkpoint'] if parent else None)
        run_command(command,attempt,'train')
        checkpoint=train/('last.pt' if model in ['326','329','336','347'] else 'best.pt')
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        record=dict(**task,checkpoint=str(checkpoint),training_dir=str(train),attempt=str(attempt))
        if 'fold' in task:
            return dict(**record,elapsed_seconds=time.monotonic()-started)
        probabilities=attempt/'probabilities'
        if model=='269': archive_to_npy(train/'valid_probs.npz',probabilities)
        elif model=='123': ema_probabilities(checkpoint,probabilities)
        else:
            script=ROOT/'source/experiments/model336_hrnet_best/evaluate_image_mean.py'
            evaluation=attempt/'hrnet_eval'
            run_command([sys.executable,'-B','-u',str(script),'--checkpoint',str(checkpoint),
                         '--data-root',str(BASE/'data'),'--split','valid','--output-dir',str(evaluation),
                         '--device','cuda:0','--batch-size','2'],attempt,'predict')
            archive_to_npy(evaluation/'valid_probabilities.npz',probabilities,True)
        selection[model]=str(probabilities); replacements[model]=[str(checkpoint)]
        if parent:
            selection['336']=parent['selection']['336']
            replacements['336']=[parent['checkpoint']]
            record['parent_id']=parent['id']; record['parent_seed']=parent['seed']
    else:
        members=[read_json(run/'completed'/f'{name}.json') for name in task['deps']]
        members.sort(key=lambda r:r['fold'])
        predictions=attempt/'b4_eval'
        script=ROOT/'source/experiments/model326_hybrid_fullstats/KoMaP/predict.py'
        run_command([sys.executable,'-B','-u',str(script),'--data',str(BASE/'data'),'--split','valid',
                     '--runs',*[m['training_dir'] for m in members], '--out',str(predictions),'--save_probs'],attempt,'predict')
        probabilities=attempt/'probabilities'
        archive_to_npy(predictions/'probabilities.npz',probabilities,True)
        selection[model]=str(probabilities); replacements[model]=[m['checkpoint'] for m in members]
        record=dict(**task,seeds=[m['seed'] for m in members],attempt=str(attempt))
    result=evaluate(selection,attempt/'evaluation',comparisons=True)
    return dict(**record,image_miou=result['image_miou'],selection=selection,replacements=replacements,
                evaluation=str(attempt/'evaluation'),elapsed_seconds=time.monotonic()-started)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--task',required=True)
    args=parser.parse_args(); run=args.run.resolve()
    if os.environ.get('MODEL349_USER_STARTED')!='1':
        raise RuntimeError('Use the user-started run.sh train --execute command')
    task=next(t for t in read_json(run/'plan.json')['tasks'] if t['id']==args.task)
    output=run/'completed'/f'{task["id"]}.json'
    if output.exists(): raise FileExistsError(output)
    if task['kind']!='select':
        import torch
        torch.set_num_threads(2)
        if torch.cuda.device_count()!=1: raise RuntimeError('Each worker must see exactly one GPU')
        print(f'GPU={os.environ.get("CUDA_VISIBLE_DEVICES")} {torch.cuda.get_device_name(0)}',flush=True)
    result=execute_task(run,task)
    write_json(output,result)
    print(f'COMPLETE {task["id"]} image_miou={result.get("image_miou")}',flush=True)

if __name__=='__main__': main()
