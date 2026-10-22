"""Sequential, single-process CLI: unlike hundred_v1's cli.py this never spawns a
torch.distributed.run subprocess or uses a file lock -- training always runs in-process, one model
at a time (per the approved plan's "sequential, single GPU" requirement)."""
import argparse
import shutil
import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from alphase.data.audit import verify_manifest
from alphase.utils import json_hash, write_json
from evoacclab.catalog import catalog
from evoacclab.workspace import ROOT, config, output_path, run_dir, verify


def preflight():
    print('보존 파일 확인:', verify(), flush=True)
    entries = catalog()
    fingerprints = set()
    folder = ROOT / 'configs/resolved'
    folder.mkdir(exist_ok=True)
    for name, item in entries.items():
        cfg = config(name)
        effective = {key: cfg[key] for key in ('model', 'train', 'loss', 'augmentation', 'sampling')}
        effective['parent'] = cfg['hundred']['parent']
        digest = json_hash(effective)
        if digest in fingerprints:
            raise ValueError('Duplicate effective configuration: ' + name)
        fingerprints.add(digest)
        target = folder / (name + '.yaml')
        serialized = OmegaConf.to_yaml(OmegaConf.create(cfg))
        if target.exists() and target.read_text() != serialized:
            raise ValueError(f'Exported configuration changed: {target}')
        if not target.exists():
            target.write_text(serialized)
    fingerprint = verify_manifest(config(next(iter(entries)))['data']['manifest'])
    free = shutil.disk_usage(ROOT).free / 2 ** 30
    print(f'{len(entries)}개 서로 다른 설정(누적 기법 1~8단계) 확인. 사용 가능 저장 공간 {free:.1f} GiB. 데이터 SHA256 {fingerprint}', flush=True)
    return dict(experiments=len(entries), dataset_sha256=fingerprint, free_gib=free,
                visible_gpus=torch.cuda.device_count())


def launch_train(name, smoke=False, resume=False, stop_after=None):
    cfg = config(name, smoke)
    folder = run_dir(name, smoke)
    from evoacclab.trainer import train
    return train(cfg, split='official', run=folder, resume=folder / 'last.pt' if resume else None, stop_after=stop_after)


def inspect_run(name, smoke, dataset_hash):
    folder = run_dir(name, smoke)
    last = folder / 'last.pt'
    if not last.is_file():
        if folder.exists() and any(folder.iterdir()):
            backup = folder.with_name(folder.name + f'.incomplete-{time.time_ns()}')
            folder.rename(backup)
            print(f'체크포인트 이전 부분 결과 보존: {backup}', flush=True)
        return False, False
    state = torch.load(last, map_location='cpu', weights_only=False, mmap=True)
    if state['config_hash'] != json_hash(config(name, smoke)) or state['dataset_hash'] != dataset_hash:
        raise ValueError('Stored run configuration/source/data differ: ' + name)
    return bool(state['completed']), True


def suite(args):
    from evoacclab.evaluation import evaluate_run
    from evoacclab.report import report
    if len(set(args.experiments)) != len(args.experiments):
        raise ValueError('Duplicate experiment IDs')
    info = preflight()
    failures, completed, started = {}, [], time.monotonic()
    status_path = output_path('smoke_suite_status.json' if args.smoke else 'suite_status.json')
    with tqdm(total=len(args.experiments), desc='Sequential model experiments', unit='model', dynamic_ncols=True) as bar:
        try:
            for name in args.experiments:
                bar.set_postfix_str(name)
                write_json(status_path, dict(state='running', active=name, completed=completed, failures=failures, total=len(args.experiments)))
                try:
                    done, has_checkpoint = inspect_run(name, args.smoke, info['dataset_sha256'])
                    if not done:
                        launch_train(name, args.smoke, resume=has_checkpoint)
                    else:
                        tqdm.write(f'이미 학습 완료: {name}; 검사 후 재사용 (오류 아님)')
                    for d4 in (False, True):
                        evaluate_run(name, args.smoke, d4)
                    completed.append(name)
                except (RuntimeError, ValueError, OSError) as exc:
                    failures[name] = f'{type(exc).__name__}: {exc}'
                    tqdm.write(f'실험 실패 기록 후 다음으로 진행: {name}: {exc}')
                report(failures, args.smoke, publish=False)
                bar.update(1)
        finally:
            _, all_done = report(failures, args.smoke, publish=True)
            state = 'completed' if all_done else ('completed_with_failures' if failures else 'partial')
            write_json(status_path, dict(state=state, completed=completed, failures=failures,
                       total=len(args.experiments), elapsed_seconds=time.monotonic() - started, terminal_only=True))
    if failures:
        raise RuntimeError(f'{len(failures)}개 미완료. 실패 목록은 보고서에 기록했습니다. 같은 명령으로 재개하세요.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='8 cumulative evolve-technique experiments on the accuracy_v1 baseline parent, sequential single-GPU training')
    sub = parser.add_subparsers(dest='command', required=True)
    names = list(catalog())
    p = sub.add_parser('suite')
    p.add_argument('--experiments', choices=names, nargs='+', default=names)
    p.add_argument('--smoke', action='store_true')
    p = sub.add_parser('train')
    p.add_argument('experiment', choices=names)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int)
    p = sub.add_parser('evaluate')
    p.add_argument('experiment', choices=names)
    p.add_argument('--d4', action='store_true')
    p.add_argument('--smoke', action='store_true')
    sub.add_parser('preflight')
    sub.add_parser('verify')
    sub.add_parser('build-hard-maps')
    p = sub.add_parser('report')
    p.add_argument('--smoke', action='store_true')
    sub.add_parser('list')
    args = parser.parse_args(argv)
    try:
        if args.command == 'suite':
            suite(args)
        elif args.command == 'train':
            if args.stop_after is not None and args.stop_after < 1:
                raise ValueError('stop-after must be positive')
            launch_train(args.experiment, args.smoke, args.resume, args.stop_after)
        elif args.command == 'evaluate':
            from evoacclab.evaluation import evaluate_run
            evaluate_run(args.experiment, args.smoke, args.d4)
        elif args.command == 'preflight':
            print(preflight())
        elif args.command == 'verify':
            print(verify())
        elif args.command == 'build-hard-maps':
            from evoacclab.mining import build_hard_maps
            build_hard_maps()
        elif args.command == 'report':
            from evoacclab.report import report
            report(smoke=args.smoke, publish=True)
        else:
            for name, item in catalog().items():
                print(name, item['group'], item['description'])
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')
