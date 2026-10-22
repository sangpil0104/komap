"""One task on one physical GPU, called only by the user-started scheduler."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, BASE, HIGH, FOLDS, read_json, write_json, environment, weight_relative, fp16_checkpoint

def set_option(argv,flag,value):
    if flag in argv: argv[argv.index(flag)+1] = str(value)
    else: argv.extend([flag,str(value)])

def command_for(task,destination):
    model = task['model']
    recipe_id = ('train_'+model if model.startswith('restart') else 'train_model'+model)
    if 'fold' in task: recipe_id += f'_fold{task["fold"]}'
    template = next(t for t in read_json(ROOT/'original_recipe.json')['tasks'] if t['id']==recipe_id)
    argv = [s.format(source=str(ROOT/'source'),run=str(BASE),workers='4') for s in template['argv']]
    set_option(argv,'--seed',task['seed'])
    set_option(argv,'--out' if 'fold' in task else '--output-dir',destination)
    if 'fold' in task: argv.append('--deterministic_workers')
    return [sys.executable,'-B','-u']+argv

def new_attempt(run,identifier):
    job = run/'jobs'/identifier; job.mkdir(parents=True,exist_ok=True)
    index = 1
    while True:
        attempt = job/f'attempt_{index}'
        try: attempt.mkdir(); return attempt
        except FileExistsError: index += 1

def execute_task(run,task):
    from engine import predict, evaluate, artifact_hashes
    attempt = new_attempt(run,task['id']); started = time.monotonic()
    model = task['model']
    record = dict(**task,attempt=str(attempt))
    if task['kind']=='baseline':
        probabilities = predict(model,attempt/'inference')
        record.update(probabilities=str(probabilities),artifacts=artifact_hashes([probabilities]))
    elif task['kind']=='select':
        result = evaluate(run,{},attempt/'evaluation',comparisons=True)
        record.update(image_miou=result['image_miou'],evaluation=str(attempt/'evaluation'),
                      artifacts=artifact_hashes([attempt/'evaluation/metrics.json']))
    else:
        replacements = {}
        if task['kind']=='train':
            train = attempt/'training'
            command = command_for(task,train)
            write_json(attempt/'train_command.json',command)
            print('COMMAND:', ' '.join(command),flush=True)
            subprocess.run(command,cwd=attempt,env=environment(),check=True)
            checkpoint = train/('last.pt' if model in ['326','329','336'] else 'best.pt')
            relative = str(weight_relative(model,task.get('fold')))
            converted = attempt/'inference_checkpoint'/relative
            conversion = fp16_checkpoint(checkpoint,converted)
            write_json(attempt/'conversion.json',conversion)
            replacements[relative] = str(converted)
            record.update(checkpoint=str(checkpoint),fp16_checkpoint=str(converted),
                          training_dir=str(train),replacements=replacements)
            if 'fold' in task:
                record.update(artifacts=artifact_hashes([checkpoint,converted,train/'args.json']),
                              elapsed_seconds=time.monotonic()-started)
                return record
        else:
            members = [read_json(run/'completed'/f'{name}.json') for name in task['deps']]
            members.sort(key=lambda x:x['fold'])
            if [m['fold'] for m in members] != FOLDS[model]: raise ValueError('Incomplete fold set')
            for member in members: replacements.update(member['replacements'])
            record.update(seeds=[m['seed'] for m in members],replacements=replacements)
        probabilities = predict('base' if model in HIGH else model,attempt/'inference',
                                replacements,family=model if model in HIGH else None)
        selection = {model:str(probabilities)}
        result = evaluate(run,selection,attempt/'evaluation',comparisons=True)
        record.update(image_miou=result['image_miou'],selection=selection,evaluation=str(attempt/'evaluation'),
                      artifacts=artifact_hashes([*replacements.values(),probabilities,attempt/'evaluation/metrics.json']))
    record['elapsed_seconds'] = time.monotonic()-started
    return record

def main():
    p = argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--task',required=True)
    args = p.parse_args()
    if os.environ.get('LITE12_USER_STARTED')!='1': raise RuntimeError('Use the user-started run.sh train --execute command')
    if Path(sys.prefix).name!='komap': raise RuntimeError('Conda komap is required')
    run = args.run.resolve()
    task = next(t for t in read_json(run/'plan.json')['tasks'] if t['id']==args.task)
    output = run/'completed'/f'{task["id"]}.json'
    if output.exists(): raise FileExistsError(output)
    if task['kind']!='select':
        import torch
        torch.set_num_threads(2)
        if torch.cuda.device_count()!=1: raise RuntimeError('Worker must see exactly one GPU')
        print(f'GPU={os.environ.get("CUDA_VISIBLE_DEVICES")} {torch.cuda.get_device_name(0)}',flush=True)
    result = execute_task(run,task)
    write_json(output,result)
    print(f'COMPLETE {task["id"]} image_miou={result.get("image_miou")}',flush=True)

if __name__=='__main__': main()
