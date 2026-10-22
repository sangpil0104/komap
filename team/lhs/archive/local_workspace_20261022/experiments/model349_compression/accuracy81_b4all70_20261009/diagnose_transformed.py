"""Fixed 400-update B4 continuation; all70 train images, frozen BN buffers."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1')
from pathlib import Path
import argparse, json, time, random, subprocess
import numpy as np
import torch
import cv2
from data import ROOT, SOURCE, SEED, Dataset, b4, sha, write, objective

def main(fold, smoke):
    torch.set_num_threads(2); cv2.setNumThreads(2)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    start=time.monotonic()
    parent=SOURCE/f'weights/model329_fullstats_width4000_uniform_group_oof/fold{fold}'
    before=sha(parent/'last.pt'); args=json.loads((parent/'args.json').read_text())
    assert args=={'arch':'unet','encoder':'efficientnet-b4','norm':'image_global','scale':1.0,'target_width':4000,'input':'zcdf3','feat_stats':None}
    model=b4.build_model(args['arch'],args['encoder'],pretrained=False)
    model.load_state_dict(torch.load(parent/'last.pt',map_location='cpu',weights_only=False),strict=True)
    model.cuda().train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    bn={n:b.detach().cpu().clone() for n,b in model.named_buffers()}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-2)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=400)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    folder=ROOT/'training'/f'fold{fold}'
    if not smoke:folder.mkdir(parents=True,exist_ok=False)
    dataset=Dataset()
    if dataset and not smoke:
        write(folder/'training_manifest.json',{'train_images':70,'rows':dataset.rows,'valid_read':False,'test_read':False,'sampling':'uniform images','sample_seed':SEED})
        write(folder/'recipe.json',{'fold':fold,'parent_sha256':before,'updates':400,'crop':1536,'batch':2,'lr':1e-5,'weight_decay':1e-2,'seed':SEED,'bn_buffers_frozen':True,'script_sha256':sha(Path(__file__)),'data_sha256':sha(ROOT/'data.py'),'valid_read':False,'test_read':False})
    history=[];samples=[];updates=0
    for step in range(1,2 if smoke else 401):
        assert time.monotonic()-start<2400,'fold training budget exceeded'
        apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        assert len(apps)<=1,'Foreign compute on GPU3; stop our training only'
        if smoke:
            batch=[dataset.sample(j) for j in range(2)]
            image=torch.stack([x[0] for x in batch]).cuda();target=torch.stack([x[1] for x in batch]).cuda()
        else:
            batch=[dataset.sample(2*(step-1)+j) for j in range(2)];samples.extend(x[2] for x in batch)
            image=torch.stack([x[0] for x in batch]).cuda();target=torch.stack([x[1] for x in batch]).cuda()
        torch.manual_seed(SEED+step);torch.cuda.manual_seed_all(SEED+step)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(image)
        assert logits.shape==(2,4,1536,1536)
        value=objective(logits,target);assert torch.isfinite(value)
        scaler.scale(value).backward();scaler.unscale_(opt)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
        old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
        assert scaler.get_scale()>=old_scale,'GradScaler skipped update'
        scheduler.step();updates+=1
        if step<=5 or step%25==0:
            row={'fold':fold,'step':step,'loss':value.item(),'seconds':time.monotonic()-start,'lr':scheduler.get_last_lr()[0]};history.append(row)
            if not smoke:write(folder/'history.json',history)
            print(json.dumps(row),flush=True)
        if not smoke and step%100==0:
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'samples':samples},folder/'resume.pt')
    assert all(torch.equal(b.detach().cpu(),bn[n]) for n,b in model.named_buffers())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert sha(parent/'last.pt')==before
    if smoke:
        torch.cuda.synchronize()
        write(ROOT/'transformed_smoke.json',{'passed':True,'input_kind':'transformed_train70_first_two','train_read':True,'synthetic_input':False,'crop':1536,'batch':2,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-start,'bn_buffers_unchanged':True,'parent_unchanged':True,'valid_read':False,'test_read':False})
        return
    assert updates==400 and len({s['image'] for s in samples})==70
    checkpoint=folder/'last.pt'
    torch.save({n:v.detach().cpu() for n,v in model.state_dict().items()},checkpoint)
    write(folder/'args.json',args);write(folder/'sample_sequence.json',samples)
    write(folder/'completion.json',{'finished':True,'fold':fold,'updates':updates,'sample_sequence_sha256':sha(folder/'sample_sequence.json'),'train_images_seen':70,'elapsed_seconds':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint_sha256':sha(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'parent_unchanged':True,'bn_buffers_unchanged':True,'all_parameters_finite':True,'valid_read':False,'test_read':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=[1,3,4],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.fold,a.smoke)
