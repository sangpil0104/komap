"""One task per allocated GPU; source and completed attempts are never overwritten."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,read_json,write_json,records,new_attempt,artifacts,command_for,environment,require_komap

def execute(run,task):
    attempt=new_attempt(run,task['id']);started=time.monotonic();done=records(run);paths=[]
    record=dict(**task,attempt=str(attempt))
    def call(command,name):
        write_json(attempt/(name+'_command.json'),command)
        print('COMMAND:',' '.join(command),flush=True)
        env=environment(ROOT)
        if task['group']=='inference':env.pop('PYTHONPATH',None)
        subprocess.run(command,check=True,cwd=attempt,env=env)
    if task['kind']=='train':
        out=attempt/'training';command=command_for(task['group'],out);call(command,'train')
        args=read_json(out/'args.json')
        assert args['iters']==21000 and args['seed']==0 and args['train_splits']=='train' and args['no_valid_eval']
        assert not (out/'valid.json').exists() and not args.get('init_checkpoint')
        checkpoint=out/'last.pt';assert checkpoint.is_file()
        record.update(checkpoint=str(checkpoint),training_dir=str(out));paths=[out,attempt/'train_command.json']
    elif task['id']=='package':
        from package_models import package_pair
        models=package_pair(attempt/'models',{c:done[f'train_{c}']['checkpoint'] for c in ['322','326','329']})
        record['models']=models;paths=[attempt/'models']
    elif task['group']=='inference':
        model=Path(done['package']['models'][task['variant']]);out=attempt/'inference'
        command=[sys.executable,'-B','-u',str(model/'compact/infer_compact.py'),
                 '--config',str(model/'compact/config.json'),'--parent-package',str(model),'--assets',str(model/'assets'),
                 '--input',str(ROOT/f'data/{task["split"]}/images'),'--output',str(out),'--gpus',os.environ['CUDA_VISIBLE_DEVICES']]
        call(command,'infer')
        expected=20 if task['split']=='valid' else 10
        assert len(list((out/'fusion/predicted_labels').glob('*.png')))==expected
        assert read_json(out/'completion.json')['gt_read'] is False
        record['inference']=str(out);paths=[out,attempt/'infer_command.json']
    elif task['id']=='export':
        from export_results import export
        result=export(run,attempt,done);record['result']=str(result);paths=[result]
    else:raise ValueError(task)
    record.update(artifacts=artifacts(paths),elapsed_seconds=time.monotonic()-started)
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--task',required=True);a=p.parse_args()
    require_komap()
    if os.environ.get('R70_USER_STARTED')!='1':raise RuntimeError('Use the user-started run.sh train --execute')
    task=next(t for t in read_json(a.run/'plan.json')['tasks'] if t['id']==a.task)
    if task['kind']!='cpu':
        import torch
        assert torch.cuda.device_count()==1 and os.environ['KOMAP_PHYSICAL_GPU'] in ['4','5','6','7']
    out=a.run/'completed'/f'{task["id"]}.json'
    if out.exists():raise FileExistsError(out)
    result=execute(a.run,task);write_json(out,result);print(f'COMPLETE {task["id"]}',flush=True)

if __name__=='__main__':main()
