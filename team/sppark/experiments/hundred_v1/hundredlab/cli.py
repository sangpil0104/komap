import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from alphase.data.audit import verify_manifest
from alphase.utils import json_hash, write_json
from hundredlab.catalog import catalog
from hundredlab.workspace import ROOT, PROJECT, OUTPUTS, config, output_path, read_json, run_dir, verify


@contextmanager
def lock(name):
    path=output_path('locks',name+'.lock');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Another process is using {name}') from None
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def controller(gpus=None):
    if int(os.environ.get('WORLD_SIZE',1))!=1:raise ValueError('Launch this command once; workers are started automatically')
    if gpus is not None and torch.cuda.device_count()<gpus:raise ValueError(f'Requested {gpus} GPUs, visible {torch.cuda.device_count()}')


def preflight(gpus=None):
    controller(gpus)
    print('기존 필수 파일 보존 확인:',verify(),flush=True)
    entries=catalog();fingerprints=set()
    folder=ROOT/'configs/resolved';folder.mkdir(exist_ok=True)
    for name,item in entries.items():
        cfg=config(name)
        effective={key:cfg[key] for key in ('model','train','loss','augmentation','sampling')}
        effective['parent']=cfg['hundred']['parent'];effective['hard']=cfg['hundred']['hard']
        digest=json_hash(effective)
        if digest in fingerprints:raise ValueError('Duplicate effective configuration: '+name)
        fingerprints.add(digest)
        target=folder/(name+'.yaml')
        serialized=OmegaConf.to_yaml(OmegaConf.create(cfg))
        if target.exists() and target.read_text()!=serialized:
            raise ValueError(f'Exported configuration changed: {target}')
        if not target.exists():target.write_text(serialized)
    fingerprint=verify_manifest(config(next(iter(entries)))['data']['manifest'])
    # Hash-verified cache is train-only; no valid/test labels are mined.
    hard=next(config(n) for n,v in entries.items() if v.get('hard'))
    maps=read_json(Path(hard['hundred']['mining_folder'])/'manifest.json')
    from alphase.data.splits import select_rows
    train,valid=select_rows(hard,'official')
    if set(maps)!={r['image_id'] for r in train}:raise ValueError('Hard maps must contain exactly the train images')
    free=shutil.disk_usage(ROOT).free/2**30
    print(f'100개 서로 다른 설정 확인. 사용 가능 저장 공간 {free:.1f} GiB. 데이터 SHA256 {fingerprint}',flush=True)
    return dict(experiments=100,groups={'recommended1':1,'local19':19,'explore80':80},
                dataset_sha256=fingerprint,free_gib=free,visible_gpus=torch.cuda.device_count())


def launch_train(args):
    cfg=config(args.experiment,args.smoke);folder=run_dir(args.experiment,args.smoke)
    if args.worker:
        from hundredlab.trainer import train
        return train(cfg,split='official',run=folder,resume=folder/'last.pt' if args.resume else None,stop_after=args.stop_after)
    controller(args.gpus)
    with lock('train-'+args.experiment+('-smoke' if args.smoke else '')):
        # Release the controller's previous inference cache before GPU workers start.
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        if folder.exists() and any(folder.iterdir()) and not args.resume:raise FileExistsError(f'{folder}: use --resume')
        command=[sys.executable,'-B','-m','torch.distributed.run','--standalone',f'--nproc_per_node={args.gpus}',
            str(ROOT/'run.py'),'train',args.experiment,'--worker','--gpus',str(args.gpus)]
        if args.smoke:command.append('--smoke')
        if args.resume:command.append('--resume')
        if args.stop_after:command+=['--stop-after',str(args.stop_after)]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.setdefault('OMP_NUM_THREADS','4')
        subprocess.run(command,env=env,check=True,cwd=ROOT)


def inspect_run(name,smoke,gpus,dataset_hash):
    folder=run_dir(name,smoke);last=folder/'last.pt'
    if not last.is_file():
        if folder.exists() and any(folder.iterdir()):
            backup=folder.with_name(folder.name+f'.incomplete-{time.time_ns()}')
            folder.rename(backup)
            print(f'체크포인트 이전 부분 결과 보존: {backup}',flush=True)
        return False,False
    state=torch.load(last,map_location='cpu',weights_only=False,mmap=True)
    if state['config_hash']!=json_hash(config(name,smoke)) or state['dataset_hash']!=dataset_hash:
        raise ValueError('Stored run configuration/source/data differ: '+name)
    if not state['completed'] and state['world_size']!=gpus:raise ValueError('Resume requires the original GPU count')
    return bool(state['completed']),True


def suite(args):
    from hundredlab.evaluation import evaluate_run
    from hundredlab.report import report
    controller(args.gpus)
    if len(set(args.experiments))!=len(args.experiments):raise ValueError('Duplicate experiment IDs')
    with lock('suite-smoke' if args.smoke else 'suite'):
        info=preflight(args.gpus)
        failures={};completed=[];started=time.monotonic()
        status_path=output_path('smoke_suite_status.json' if args.smoke else 'suite_status.json')
        with tqdm(total=len(args.experiments),desc='Sequential model experiments',unit='model',dynamic_ncols=True) as bar:
            try:
                for name in args.experiments:
                    bar.set_postfix_str(name)
                    write_json(status_path,dict(state='running',active=name,completed=completed,failures=failures,total=len(args.experiments)))
                    try:
                        done,resume=inspect_run(name,args.smoke,args.gpus,info['dataset_sha256'])
                        if not done:
                            launch_train(argparse.Namespace(experiment=name,smoke=args.smoke,gpus=args.gpus,
                                worker=False,resume=resume,stop_after=None))
                        else:tqdm.write(f'이미 학습 완료: {name}; 검사 후 재사용 (오류 아님)')
                        for d4 in (False,True):evaluate_run(name,args.smoke,d4)
                        completed.append(name)
                    except (subprocess.CalledProcessError,RuntimeError,ValueError,OSError) as exc:
                        failures[name]=f'{type(exc).__name__}: {exc}'
                        tqdm.write(f'실험 실패 기록 후 다음으로 진행: {name}: {exc}')
                    report(failures,args.smoke,publish=False)
                    bar.update(1)
            finally:
                _,all_done=report(failures,args.smoke,publish=True)
                state='completed' if all_done else ('completed_with_failures' if failures else 'partial')
                write_json(status_path,dict(state=state,completed=completed,failures=failures,
                    total=len(args.experiments),elapsed_seconds=time.monotonic()-started,terminal_only=True))
        if failures:raise RuntimeError(f'{len(failures)}개 미완료. 실패 목록은 보고서에 기록했습니다. 같은 명령으로 재개하세요.')


def main(argv=None):
    parser=argparse.ArgumentParser(description='100 isolated model experiments, sequential training and automatic Markdown report')
    sub=parser.add_subparsers(dest='command',required=True);names=list(catalog())
    p=sub.add_parser('suite');p.add_argument('--gpus',type=int,choices=[1,2,4,8],default=8)
    p.add_argument('--experiments',choices=names,nargs='+',default=names);p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('train');p.add_argument('experiment',choices=names)
    p.add_argument('--gpus',type=int,choices=[1,2,4,8],default=8)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    for cmd in ('evaluate','predict'):
        p=sub.add_parser(cmd);p.add_argument('experiment',choices=names)
        p.add_argument('--d4',action='store_true');p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('report');p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('preflight');p.add_argument('--gpus',type=int,choices=[1,2,4,8])
    p=sub.add_parser('verify');p.add_argument('--full',action='store_true')
    sub.add_parser('list')
    args=parser.parse_args(argv)
    try:
        if args.command=='suite':suite(args)
        elif args.command=='train':
            if args.stop_after is not None and args.stop_after<1:raise ValueError('stop-after must be positive')
            launch_train(args)
        elif args.command in ('evaluate','predict'):
            controller()
            from hundredlab.evaluation import evaluate_run,predict_test
            (evaluate_run if args.command=='evaluate' else predict_test)(args.experiment,args.smoke,args.d4)
        elif args.command=='preflight':print(preflight(args.gpus))
        elif args.command=='verify':print(verify(args.full))
        elif args.command=='report':
            from hundredlab.report import report
            with lock('suite-smoke' if args.smoke else 'suite'):report(smoke=args.smoke,publish=True)
        else:
            for name,item in catalog().items():print(name,item['group'],item['description'])
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as exc:parser.exit(2,f'Error: {exc}\n')
