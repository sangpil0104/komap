"""One trial on eight GPUs. Invoked only by the user-started sequential runner."""
import argparse
from datetime import timedelta
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from boundarylab.data import TrainDataset, one_sample_collate
from boundarylab.inference import predict
from boundarylab.io import (capture_rng, fingerprint, immutable_torch_save, read_json,
                            require_komap, reserve_result, restore_rng, seed_all, sha256, write_json)
from boundarylab.losses import loss_for
from boundarylab.metrics import PALETTE, mask_ids, metrics
from boundarylab.models import BoundaryModel
from boundarylab.report import comparison_panel, finish_report, result_manifest


def lr_factor(step, cfg):
    return min(1.0, (step+1)/cfg['warmup']) * 0.5*(1+math.cos(math.pi*min(step,cfg['steps'])/cfg['steps']))


def latest_checkpoint(run):
    files = sorted((run/'checkpoints').glob('step_*.pt'))
    return files[-1] if files else None


def save_training_state(run, model, optimizer, scheduler, step, protocol_hash, elapsed, rank, world):
    states = [None]*world if rank == 0 else None
    dist.gather_object(capture_rng(), states, dst=0)
    if rank == 0:
        path = run/'checkpoints'/f'step_{step:06d}.pt'
        immutable_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  scheduler=scheduler.state_dict(), step=step, rng=states,
                                  protocol_hash=protocol_hash, elapsed_seconds=elapsed), path)
        write_json(run/'latest.json', dict(checkpoint=str(path), step=step, sha256=sha256(path)))
    dist.barrier()


def train_model(model, arm, plan, run, rank, world, device):
    cfg = arm['config']
    if arm['initialization'] != plan['encoder_weights']:
        raise ValueError('New training must initialize ONLY from ImageNet encoder weights')
    if world != 8 or cfg['global_batch'] != world:
        raise ValueError('Exactly eight GPUs, one crop per GPU, are required')
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=True, find_unused_parameters=False)
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],
                                  lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, cfg))
    protocol_hash = fingerprint(dict(arm=arm, source=plan['source_hash'], data=plan['data_hash'],
                                     environment=plan['environment'], encoder=plan['encoder_sha256'], world=world))
    start, previous_elapsed = 0, 0.0
    resume = latest_checkpoint(run)
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['protocol_hash'] != protocol_hash:
            raise ValueError('Resume rejected: configuration/source/data/environment/initialization changed')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, previous_elapsed = state['step'], state['elapsed_seconds']
        restore_rng(state['rng'][rank])
        del state
    else:
        # New-branch initialization must not shift the base model's stochastic-depth RNG.
        seed_all(cfg['seed'] + rank)
    rows = [r for r in plan['data'] if r['split'] == 'train']
    if start < cfg['steps']:
        dataset = TrainDataset(rows, cfg)
        sampler = range(start*world+rank, cfg['steps']*world, world)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=cfg['workers_per_rank'],
                            collate_fn=one_sample_collate, pin_memory=True,
                            persistent_workers=cfg['workers_per_rank'] > 0,
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if cfg['workers_per_rank'] else {}),
                            generator=torch.Generator().manual_seed(cfg['seed'] + rank))
        wrapped.train()
        clock = time.monotonic()
        progress = tqdm(total=cfg['steps'], initial=start, desc=arm['id'], unit='update', disable=rank != 0,
                        mininterval=10, dynamic_ncols=True)
        for step, batch in enumerate(loader, start+1):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = wrapped(batch['detail'], batch['native_gray'])
            loss, parts = loss_for(output, batch, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at {step}: {loss.item()}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), cfg['clip_grad'])
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite gradients at {step}')
            optimizer.step()
            scheduler.step()
            elapsed = previous_elapsed + time.monotonic()-clock
            if rank == 0:
                progress.update(1)
                progress.set_postfix(loss=f'{loss.item():.5f}', refresh=False)
                if step == start+1 or step % 10 == 0 or step == cfg['steps']:
                    write_json(run/'progress.json', dict(phase='train', step=step, total=cfg['steps'],
                                elapsed_seconds=elapsed, eta_seconds=(elapsed-previous_elapsed)/(step-start)*(cfg['steps']-step),
                                loss=loss.item(), detail_loss=parts['detail'].item(), native_loss=parts['native'].item()))
                if step % 100 == 0 or step == start+1:
                    with (run/'training.jsonl').open('a') as f:
                        import json
                        f.write(json.dumps(dict(step=step, loss=loss.item(), elapsed_seconds=elapsed,
                                                learning_rate=scheduler.get_last_lr()[0]))+'\n')
            if step % cfg['checkpoint_every'] == 0 or step == cfg['steps']:
                save_training_state(run, model, optimizer, scheduler, step, protocol_hash, elapsed, rank, world)
        progress.close()
        previous_elapsed = elapsed
        del loader, dataset
    if rank == 0:
        weights = run/'last.pt'
        if not weights.exists():
            immutable_torch_save(model.state_dict(), weights)
        else:
            old = torch.load(weights, map_location='cpu', weights_only=True)
            if old.keys() != model.state_dict().keys() or any(not torch.equal(old[k],v.cpu()) for k,v in model.state_dict().items()):
                raise ValueError('Existing final weights do not match resumed final checkpoint')
            del old
        write_json(run/'trained.json', dict(initialization='imagenet_encoder_and_fresh_decoder',
                   initialization_path=plan['encoder_weights'], initialization_sha256=plan['encoder_sha256'],
                   steps=cfg['steps'], samples=cfg['steps']*world, protocol_hash=protocol_hash,
                   training_seconds=previous_elapsed, weights=str(weights), weights_sha256=sha256(weights)))
    dist.barrier()
    del wrapped, optimizer, scheduler
    torch.cuda.empty_cache()
    return model


def evaluate(model, arm, plan, suite, run, rank, world, device):
    model.eval()
    result = [str(reserve_result(plan['results_root'])) if rank == 0 else None]
    dist.broadcast_object_list(result, src=0)
    result = Path(result[0])
    if rank == 0:
        for name in ('valid_mask_comparison', 'predictions', 'per_image'):
            (result/name).mkdir()
        write_json(run/'progress.json', dict(phase='evaluate', result=str(result), total=20, step=0))
        # A fresh result directory on every evaluation attempt preserves partial attempts.
        write_json(run/f'evaluation_{result.name}.json', dict(result=str(result)), exclusive=True)
    dist.barrier()
    rows = [r for r in plan['data'] if r['split'] == 'valid']
    shard = rows[rank::world]
    started = time.monotonic()
    for done, row in enumerate(shard, 1):
        gray = np.array(Image.open(row['image']).convert('L'))
        probability = predict(model, gray, arm['config'], device)
        pred = probability.argmax(0).cpu().numpy().astype(np.uint8)
        target = mask_ids(np.array(Image.open(row['mask'])))
        record = dict(image_id=row['image_id'], shape=list(target.shape), **metrics(target, pred))
        Image.fromarray(PALETTE[pred]).save(result/'predictions'/f'{row["image_id"]}.png')
        panel = comparison_panel(Image.open(row['image']), target, pred, record['image_miou'], row['image_id'])
        panel.save(result/'valid_mask_comparison'/f'{row["image_id"]}.png')
        write_json(result/'per_image'/f'{row["image_id"]}.json', record, exclusive=True)
        write_json(run/f'eval_rank_{rank}.json', dict(result=str(result), done=done, total=len(shard),
                                                    elapsed_seconds=time.monotonic()-started))
        del probability, panel
    dist.barrier()
    if rank == 0:
        records = [read_json(result/'per_image'/f'{r["image_id"]}.json') for r in rows]
        weights = Path(arm['initialization']) if arm['kind'] == 'eval' else run/'last.pt'
        provenance = dict(suite=str(suite), arm_id=arm['id'], source_hash=plan['source_hash'], data_hash=plan['data_hash'],
                          initialization=arm['initialization'], weights=str(weights), weights_sha256=sha256(weights),
                          environment=plan['environment'], gpu_ids=plan['gpus'], world_size=world,
                          evaluation_seconds=time.monotonic()-started, selection='fixed final step; no valid checkpoint selection')
        if arm['kind'] == 'train':
            provenance['training'] = read_json(run/'trained.json')
        summary = finish_report(result, arm, records, provenance)
        write_json(result/'complete.json', dict(files=result_manifest(result)), exclusive=True)
        write_json(run/'complete.json', dict(result=str(result), weights_sha256=sha256(weights),
                    image_miou=summary['image_miou'], arm_hash=fingerprint(arm)), exclusive=True)
        write_json(run/'progress.json', dict(phase='complete', result=str(result), step=20, total=20,
                                             image_miou=summary['image_miou']))
        print(f'{arm["id"]}: image mIoU={summary["image_miou"]*100:.6f}% -> {result}', flush=True)
    dist.barrier()


def main():
    require_komap()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()
    plan = read_json(args.suite/'plan.json')
    arm = next(a for a in plan['arms'] if a['id'] == args.experiment)
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    if world != 8:
        raise ValueError('This worker requires eight GPU processes')
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=40), device_id=device)
    run = args.suite/'trials'/arm['id']
    run.mkdir(parents=True, exist_ok=True)
    seed_all(arm['config']['seed'])
    if arm['kind'] == 'train':
        model = BoundaryModel(arm['config'], encoder_weights=plan['encoder_weights']).to(device)
        model = train_model(model, arm, plan, run, rank, world, device)
    else:
        model = BoundaryModel(arm['config'], reference_checkpoint=arm['initialization']).to(device)
    evaluate(model, arm, plan, args.suite, run, rank, world, device)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
