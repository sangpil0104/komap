"""GPU verification ONLY: no optimizer.step, no real-data training, no saved trained model."""
import argparse
from datetime import timedelta
import gc
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from boundarylab.config import CHECKPOINT, ENCODER_WEIGHTS, PROJECT, make_plan
from boundarylab.data import list_rows
from boundarylab.inference import predict
from boundarylab.io import require_komap, seed_all, write_json, read_json
from boundarylab.losses import ce_dice, loss_for
from boundarylab.metrics import mask_ids, metrics, summarize
from boundarylab.models import BoundaryModel
from boundarylab.report import comparison_panel


def check_global_gradient(device, rank, world):
    seed_all(101)
    model=DDP(torch.nn.Conv2d(3,4,1).to(device),device_ids=[device.index])
    gen=torch.Generator().manual_seed(500+rank)
    x=torch.randn(1,3,5,7,generator=gen).to(device)
    y=torch.randint(0,4,(1,5,7),generator=gen).to(device)
    loss=ce_dice(model(x),y)
    loss.backward()
    # Rank-zero direct global batch reference, without calling distributed loss again.
    if rank==0:
        plain=torch.nn.Conv2d(3,4,1)
        plain.load_state_dict({k:v.cpu() for k,v in model.module.state_dict().items()})
        xs=[];ys=[]
        for r in range(world):
            g=torch.Generator().manual_seed(500+r)
            xs.append(torch.randn(1,3,5,7,generator=g))
            ys.append(torch.randint(0,4,(1,5,7),generator=g))
        logits=plain(torch.cat(xs));target=torch.cat(ys)
        p=logits.softmax(1);oh=F.one_hot(target,4).permute(0,3,1,2).float()
        reference=F.cross_entropy(logits,target)+1-((2*(p*oh).sum((0,2,3))+1)/(p.sum((0,2,3))+oh.sum((0,2,3))+1)).mean()
        reference.backward()
        torch.testing.assert_close(model.module.weight.grad.cpu(),plain.weight.grad,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(loss.cpu(),reference,atol=1e-6,rtol=1e-6)
    dist.barrier()
    del model


def backward_checks(args, device, rank, world):
    check_global_gradient(device,rank,world)
    reports=[]
    for arm in make_plan(crop=args.crop):
        if arm['kind']!='train':
            continue
        cfg=arm['config']
        seed_all(32)
        model=BoundaryModel(cfg,encoder_weights=ENCODER_WEIGHTS).to(device)
        model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model=DDP(model,device_ids=[device.index],find_unused_parameters=False)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        started=time.monotonic()
        # A second forward/backward catches incomplete DDP reductions from the first.
        for repeat in range(2):
            gen=torch.Generator(device=device).manual_seed(9000+rank+repeat)
            p=args.crop
            n=max(16,round(p*1440/4000))
            detail=torch.randn(1,3,p,p,device=device,generator=gen)
            native=torch.randn(1,1,n,n,device=device,generator=gen) if cfg['geometry']=='aligned_native_roi_v1' else None
            target=torch.randint(0,4,(1,p,p),device=device,generator=gen)
            target_native=torch.randint(0,4,(1,n,n),device=device,generator=gen) if native is not None else None
            batch=dict(detail=detail,target=target,native_gray=native,native_target=target_native,
                       boundary=(target==0).float())
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(detail,native)
            loss,_=loss_for(output,batch,cfg)
            loss.backward()
            missing=[name for name,param in model.named_parameters() if param.requires_grad and param.grad is None]
            if missing:
                raise AssertionError(f'{arm["id"]}: no gradient {missing}')
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise AssertionError('Nonfinite synthetic loss/gradients')
            del batch,detail,native,target,target_native,output,loss
        torch.cuda.synchronize(device)
        info=dict(id=arm['id'],rank=rank,crop=args.crop,forward_backward_passes=2,optimizer_steps=0,
                  peak_memory_mib=torch.cuda.max_memory_allocated(device)/2**20,seconds=time.monotonic()-started)
        collected=[None]*world if rank==0 else None
        dist.gather_object(info,collected,dst=0)
        if rank==0:
            reports.append(dict(experiment=arm['id'],ranks=collected))
            print(f'{arm["id"]}: 8-GPU synthetic forward/backward passed; max {max(r["peak_memory_mib"] for r in collected):.0f} MiB',flush=True)
            write_json(args.output/'backward.json',dict(real_training_started=False,optimizer_steps=0,
                       global_dice_gradient_matches_direct_batch=True,experiments=reports))
        del model
        gc.collect();torch.cuda.empty_cache()
        dist.barrier()


def reference_checks(args, device, rank, world):
    cfg=make_plan()[0]['config']
    seed_all(32)
    model=BoundaryModel(cfg,reference_checkpoint=CHECKPOINT).to(device).eval()
    rows=list_rows(PROJECT/'dataset','valid')
    results=[]
    for row in rows[rank::world]:
        gray=np.array(Image.open(row['image']).convert('L'))
        probability=predict(model,gray,cfg,device)
        pred=probability.argmax(0).cpu().numpy().astype(np.uint8)
        target=mask_ids(np.array(Image.open(row['mask'])))
        record=dict(image_id=row['image_id'],**metrics(target,pred))
        panel=comparison_panel(Image.open(row['image']),target,pred,record['image_miou'],row['image_id'])
        panel.save(args.output/f'{row["image_id"]}.png')
        results.append(record)
        del probability,panel
    collected=[None]*world if rank==0 else None
    dist.gather_object(results,collected,dst=0)
    if rank==0:
        allrows=sorted([x for shard in collected for x in shard],key=lambda r:r['image_id'])
        summary=summarize(allrows)
        expected=read_json(PROJECT/'results/model_results_0929_1/summary.json')['winner_score_percent']/100
        summary.update(expected_image_miou=expected,absolute_difference=abs(summary['image_miou']-expected),
                       real_training_started=False,optimizer_steps=0)
        write_json(args.output/'reference.json',summary)
        print(f'Reference valid20: {summary["image_miou"]*100:.10f}%, expected {expected*100:.10f}%',flush=True)
        if abs(summary['image_miou']-expected)>1e-7:
            raise AssertionError('Archived seed32 score not reproduced')
    dist.barrier()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['backward','reference'],required=True)
    parser.add_argument('--crop',type=int,default=1536)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require_komap()
    rank,world,local=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1);torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20),device_id=device)
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    if args.mode=='backward':
        backward_checks(args,device,rank,world)
    else:
        reference_checks(args,device,rank,world)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
