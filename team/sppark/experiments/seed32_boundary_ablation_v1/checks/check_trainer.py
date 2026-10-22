"""Exercise the production DDP trainer/resume with a tiny synthetic network only.

No real images, B4 model, or segmentation checkpoint are trained by this check.
"""
import argparse
from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch import nn

import worker
from boundarylab.config import ENCODER_WEIGHTS, make_plan
from boundarylab.io import read_json, require_komap, seed_all, write_json


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, rows, cfg):
        assert rows==[], 'This check must never access real data'
        self.cfg=cfg
    def __len__(self):return self.cfg['steps']*8
    def __getitem__(self,index):
        g=torch.Generator().manual_seed(7000+int(index))
        return dict(detail=torch.randn(3,32,32,generator=g),
                    target=torch.randint(0,4,(32,32),generator=g),
                    boundary=torch.zeros(32,32),native_gray=None,native_target=None)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.BatchNorm2d(8),
                                  nn.ReLU(),nn.Dropout2d(.2),nn.Conv2d(8,4,1))
    def forward(self,x,native_gray=None):return dict(logits=self.layers(x),native=None)


class SyntheticStop(Exception):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require_komap()
    rank,world,local=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1);torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=5))
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    worker.TrainDataset=ToyDataset
    arm=deepcopy(next(a for a in make_plan(steps=4,crop=64) if a['id']=='T00_control'))
    arm['id']='SYNTHETIC_TOY_ONLY'
    arm['config'].update(checkpoint_every=2,workers_per_rank=0,warmup=2)
    plan=dict(encoder_weights=str(ENCODER_WEIGHTS),encoder_sha256='synthetic-no-encoder-loaded',
              source_hash='synthetic-check',data_hash='no-real-data',environment={},data=[])
    for name in ('uninterrupted','interrupted'):
        (args.output/name).mkdir(exist_ok=True)
    def fresh():
        seed_all(32)
        return ToyModel().to(device)
    full=worker.train_model(fresh(),arm,plan,args.output/'uninterrupted',rank,world,device)
    full_state={k:v.detach().clone() for k,v in full.state_dict().items()}
    del full
    save=worker.save_training_state
    def interrupted(*pos,**kw):
        save(*pos,**kw)
        if pos[4]==2:raise SyntheticStop('simulate interruption after checkpoint 2')
    worker.save_training_state=interrupted
    try:
        worker.train_model(fresh(),arm,plan,args.output/'interrupted',rank,world,device)
        raise AssertionError('Expected simulated interruption')
    except SyntheticStop:
        pass
    worker.save_training_state=save
    resumed=worker.train_model(fresh(),arm,plan,args.output/'interrupted',rank,world,device)
    for key,value in resumed.state_dict().items():
        torch.testing.assert_close(value,full_state[key],rtol=0,atol=0)
    dist.barrier()
    if rank==0:
        assert read_json(args.output/'interrupted'/'trained.json')['steps']==4
        write_json(args.output/'verification.json',dict(passed=True,world_size=world,
                   network='tiny synthetic CNN; NOT EfficientNet-B4',real_images_used=0,
                   interrupted_after_update=2,resumed_to_update=4,
                   final_parameters_and_bn_buffers_bitwise_equal=True,
                   real_training_started=False))
        print('Production trainer: synthetic 8-GPU interruption/resume is bitwise identical to uninterrupted execution.',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
