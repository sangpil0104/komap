"""Step 2: train every exh14 member for one (fold, seed) with the recorded Model349 recipes.

    python train_fold_seed.py --work WORK --fold K --seed S [--only stage,...] [--smoke]

Data per fold (from prepare_data.py; the fold's OOF images are absent everywhere below):
  * ResNet/ASPP chain  -> folds/foldK/sel   (train = fitting images, valid = selection split used for best.pt)
  * B4 322/326/329     -> folds/foldK/b4cv  (all fold-training images, last checkpoint, no selection)
  * HRNet 336/347      -> folds/foldK/full  (all fold-training images, last checkpoint, no selection)
Recipe seeds are offset by 1000*S, so S=0 reproduces the historical seeds. Finished stages
(marker file) are skipped; a stage directory without a marker is never overwritten.
--smoke shrinks epochs/iterations to check the code path only (scores are meaningless).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import VENDOR, read_json, vendor_env, write_json

H = VENDOR / 'experiments' / 'historical_scripts'
B4_322 = VENDOR / 'experiments' / 'model322_team_b4' / 'KoMaP' / 'train.py'
B4_326 = VENDOR / 'experiments' / 'model326_hybrid_fullstats' / 'KoMaP' / 'train.py'
HR = VENDOR / 'experiments' / 'model336_hrnet_best' / 'train.py'


def stages(sel, b4, full, out, seed, cw):
    s = lambda base: str(base + 1000 * seed)
    o = lambda name: str(out / name)
    b4common = ['--arch', 'unet', '--encoder', 'efficientnet-b4', '--crop', '1536', '--bs', '2', '--iters', '16000',
                '--lr', '3e-4', '--wd', '1e-2', '--dice_w', '1', '--cldice_w', '0', '--workers', '4', '--no_valid_eval',
                '--data', str(b4), '--train_splits', 'train_all']
    return [
        # name, script, args, smoke overrides, checkpoint produced
        ('model6', H / 'train_models_2_to_6.py', ['--model-id', '6', '--data-dir', sel, '--output-dir', o('model6'),
          '--epochs', '30', '--image-size', '512', '--num-workers', '4', '--seed', s(42)], {'--epochs': '1'}, 'model6/best.pt'),
        ('model62', H / 'improve_model6.py', ['--data-dir', sel, '--checkpoint', o('model6/best.pt'), '--output-dir', o('model62'),
          '--epochs', '20', '--batch-size', '4', '--encoder-lr', '5e-6', '--decoder-lr', '5e-5', '--loss-mode', 'weighted',
          '--aug-mode', 'safe_plus', '--select-tta', '--seed', s(42)], {'--epochs': '1'}, 'model62/best.pt'),
        ('model97', H / 'improve_five_causes.py', ['--data-dir', sel, '--checkpoint', o('model62/best.pt'), '--output-dir', o('model97'),
          '--mode', 'train', '--patch-size', '512', '--patches-per-image', '6', '--val-stride', '448', '--final-stride', '384',
          '--freeze-epochs', '3', '--finetune-epochs', '12', '--encoder-lr', '5e-6', '--decoder-lr', '1e-4', '--seed', s(42)],
         {'--freeze-epochs': '1', '--finetune-epochs': '1', '--patches-per-image': '1'}, 'model97/best.pt'),
        ('model98', H / 'hard_negative_finetune.py', ['--data-dir', sel, '--checkpoint', o('model97/best.pt'), '--output-dir', o('model98'),
          '--epochs', '4', '--encoder-lr', '1e-6', '--decoder-lr', '5e-5', '--seed', s(42)], {'--epochs': '1'}, 'model98/best.pt'),
        ('model109', H / 'encoder_structure_search.py', ['--candidate', 'aspp', '--data-dir', sel, '--checkpoint', o('model98/best.pt'),
          '--output-dir', o('model109'), '--epochs', '4', '--patch-size', '512', '--val-stride', '384', '--patches-per-image', '6',
          '--lr', '1.5e-5', '--eval-bias', '-0.2', '--seed', s(42), '--skip-test-inference'],
         {'--epochs': '1', '--patches-per-image': '1'}, 'model109/best.pt'),
        ('restart51', H / 'encoder_structure_search.py', ['--candidate', 'aspp', '--data-dir', sel, '--checkpoint', o('model109/best.pt'),
          '--output-dir', o('restart51'), '--patch-size', '768', '--val-stride', '576', '--tile-batch-size', '1',
          '--patches-per-image', '4', '--epochs', '6', '--batch-size', '2', '--lr', '8e-6', '--eval-bias', '-0.55',
          '--seed', s(123), '--skip-test-inference'], {'--epochs': '1', '--patches-per-image': '1'}, 'restart51/best.pt'),
        ('restart59', H / 'encoder_structure_search.py', ['--candidate', 'aspp', '--data-dir', sel, '--checkpoint', o('model109/best.pt'),
          '--output-dir', o('restart59'), '--patch-size', '768', '--val-stride', '576', '--tile-batch-size', '1',
          '--patches-per-image', '4', '--epochs', '5', '--batch-size', '2', '--lr', '8e-6', '--eval-bias', '-0.55',
          '--seed', s(456), '--skip-test-inference'], {'--epochs': '1', '--patches-per-image': '1'}, 'restart59/best.pt'),
        ('restart61', H / 'pairwise_margin_finetune.py', ['--data-dir', sel, '--checkpoint', o('restart51/best.pt'),
          '--output-dir', o('restart61'), '--batch-size', '1', '--num-workers', '2', '--seed', s(42)], {'--epochs': '1'},
         'restart61/best.pt'),
        ('model194', H / 'model194_multistage_team_schedule.py', ['--data-dir', sel, '--checkpoint', o('model109/best.pt'),
          '--output-dir', o('model194'), '--stage0-epochs', '3', '--stage1-epochs', '3', '--stage2-epochs', '0',
          '--stage0-patches', '16', '--stage1-patches', '16', '--stage2-patches', '8', '--stage0-batch', '2', '--stage1-batch', '2',
          '--stage2-batch', '1', '--encoder-lr', '1e-4', '--decoder-lr', '3e-4', '--selection-metric', 'image_mean_miou_exclude',
          '--seed', s(2028)], {'--stage0-epochs': '1', '--stage1-epochs': '1', '--stage0-patches': '1', '--stage1-patches': '1'},
         'model194/best.pt'),
        ('model195', H / 'model194_multistage_team_schedule.py', ['--data-dir', sel, '--checkpoint', o('model194/best.pt'),
          '--output-dir', o('model195'), '--stage0-epochs', '0', '--stage1-epochs', '0', '--stage2-epochs', '2',
          '--stage2-patches', '8', '--stage2-batch', '2', '--eval-patch-size', '768', '--eval-stride', '576',
          '--tile-batch-size', '2', '--encoder-lr', '5e-5', '--decoder-lr', '1.5e-4', '--weight-decay', '1e-4',
          '--selection-metric', 'image_mean_miou_exclude', '--seed', s(2028), '--num-workers', '4'],
         {'--stage2-epochs': '1', '--stage2-patches': '1'}, 'model195/best.pt'),
        ('model239', H / 'model239_native_stem_finetune.py', ['--data-dir', sel, '--checkpoint', o('model195/best.pt'),
          '--output-dir', o('model239'), '--epochs', '4', '--encoder-lr', '1e-5', '--decoder-lr', '3e-5', '--native-lr', '3e-4',
          '--seed', s(239)], {'--epochs': '1'}, 'model239/best.pt'),
        ('model242', H / 'model242_ema_continuation.py', ['--data-dir', sel, '--checkpoint', o('model239/best.pt'),
          '--output-dir', o('model242'), '--epochs', '4', '--ema-decay', '0.99', '--encoder-lr', '5e-6', '--decoder-lr', '1.5e-5',
          '--native-lr', '1e-4', '--seed', s(242)], {'--epochs': '1'}, 'model242/best_ema.pt'),
        ('model257', H / 'model253_highres_scale_finetune.py', ['--data-dir', sel, '--checkpoint', o('model242/best_ema.pt'),
          '--output-dir', o('model257'), '--scale', '1', '--patch-size', '1024', '--patches-per-image', '8', '--batch-size', '2',
          '--epochs', '4', '--scheduler-epochs', '4', '--encoder-lr', '5e-6', '--decoder-lr', '1.5e-5', '--native-lr', '1e-4',
          '--weight-decay', '1e-4', '--val-stride', '576', '--tta', 'd4', '--selection-metric', 'image_mean_miou',
          '--seed', s(257), '--global-representation', '--num-workers', '4'],
         {'--epochs': '1', '--scheduler-epochs': '1', '--patches-per-image': '1'}, 'model257/best.pt'),
        ('model269', H / 'model253_highres_scale_finetune.py', ['--data-dir', sel, '--checkpoint', o('model257/best.pt'),
          '--output-dir', o('model269'), '--scale', '1', '--patch-size', '1024', '--patches-per-image', '8', '--batch-size', '2',
          '--epochs', '4', '--scheduler-epochs', '4', '--encoder-lr', '2.5e-6', '--decoder-lr', '7.5e-6', '--native-lr', '5e-5',
          '--weight-decay', '1e-4', '--val-stride', '576', '--tta', 'd4', '--selection-metric', 'image_mean_miou',
          '--seed', s(269), '--global-representation', '--num-workers', '4'],
         {'--epochs': '1', '--scheduler-epochs': '1', '--patches-per-image': '1'}, 'model269/best.pt'),
        ('b4_322', B4_322, ['--out', o('b4_322'), '--norm', 'imagenet', '--input', 'gray3', '--scale', '3',
          '--target_width', '0', '--sample', 'area', '--seed', s(0)] + b4common, {'--iters': '30'}, 'b4_322/last.pt'),
        ('b4_326', B4_326, ['--out', o('b4_326'), '--norm', 'image_global', '--input', 'zcdf3', '--scale', '3',
          '--target_width', '0', '--sample', 'area', '--seed', s(0), '--deterministic_workers'] + b4common,
         {'--iters': '30'}, 'b4_326/last.pt'),
        ('b4_329', B4_326, ['--out', o('b4_329'), '--norm', 'image_global', '--input', 'zcdf3', '--scale', '1',
          '--target_width', '4000', '--sample', 'uniform', '--seed', s(0), '--deterministic_workers'] + b4common,
         {'--iters': '30'}, 'b4_329/last.pt'),
        ('hrnet336', HR, ['--data-root', full, '--output-dir', o('hrnet336'), '--batch-size', '4', '--accumulation-steps', '2',
          '--epochs', '70', '--samples-per-image', '12', '--lr', '3e-4', '--encoder-lr-mult', '1.0', '--weight-decay', '1e-4',
          '--boundary-head-weight', '1.0', '--seed', s(0), '--val-every', '0'],
         {'--epochs': '1', '--samples-per-image': '1'}, 'hrnet336/last.pt'),
        ('hrnet347', HR, ['--data-root', full, '--output-dir', o('hrnet347'), '--batch-size', '4', '--accumulation-steps', '2',
          '--epochs', '12', '--samples-per-image', '12', '--lr', '1e-4', '--encoder-lr-mult', '0.3', '--weight-decay', '1e-4',
          '--boundary-head-weight', '1.0', '--seed', s(347), '--val-every', '0', '--init-checkpoint', o('hrnet336/last.pt'),
          '--ce-class-weights', *[f'{w:.8f}' for w in cw]], {'--epochs': '1', '--samples-per-image': '1'}, 'hrnet347/last.pt'),
    ]


def apply_smoke(args, overrides):
    args = list(args)
    for flag, value in overrides.items():
        if flag in args:
            args[args.index(flag) + 1] = value
        else:
            args += [flag, value]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--only', default='', help='comma-separated stage names (default: all)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args()
    base = a.work / 'folds' / f'fold{a.fold}'
    if not (base / 'calibration.json').is_file():
        sys.exit(f'run prepare_data.py first: {base}')
    out = a.work / 'runs' / f'fold{a.fold}' / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    cw = read_json(base / 'class_weights.json')['weights']
    todo = stages(str(base / 'sel'), str(base / 'b4cv'), str(base / 'full'), out, a.seed, cw)
    only = set(filter(None, a.only.split(',')))
    env = vendor_env()
    for name, script, args, smoke, ckpt in todo:
        if only and name not in only:
            continue
        marker = out / name / 'STAGE_DONE.json'
        if marker.exists():
            print(f'skip {name} (done)', flush=True)
            continue
        if (out / name).exists() and any((out / name).iterdir()):
            sys.exit(f'{out / name} exists without STAGE_DONE.json; keep it for inspection and use a new --work')
        (out / name).mkdir(parents=True, exist_ok=True)
        cmd = [a.python, '-u', str(script)] + (apply_smoke(args, smoke) if a.smoke else args)
        print(f'== fold{a.fold} seed{a.seed} {name}', flush=True)
        t0 = time.time()
        with (out / name / 'stage.log').open('w') as log:
            code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(script.parent))
        if code or not (out / ckpt).is_file():
            sys.exit(f'stage {name} failed (exit {code}); see {out / name / "stage.log"}')
        write_json(marker, {'stage': name, 'command': cmd, 'seconds': time.time() - t0, 'smoke': a.smoke,
                            'checkpoint': ckpt})
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
