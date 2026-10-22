"""Additional frozen-network inference factors; saved model weights are untouched."""
import argparse
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import run as runner


class ForwardAdapter(nn.Module):
    def __init__(self, model, specification):
        super().__init__();self.model=model;self.specification=specification

    def forward(self,x):
        cfg=self.specification
        if 'temperature' in cfg:return self.model(x)/cfg['temperature']
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        image=x*std+mean
        if cfg.get('quantize_input'):image=image.clamp(0,1).mul(255).round().div(255)
        if cfg.get('clip_input'):image=image.clamp(0,1)
        if 'input_blur' in cfg:
            sigma=cfg['input_blur'];radius=max(1,round(3*sigma));axis=torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype)
            kernel=torch.exp(-axis.square()/(2*sigma*sigma));kernel/=kernel.sum();kernel=(kernel[:,None]*kernel[None,:])[None,None].repeat(3,1,1,1)
            image=F.conv2d(F.pad(image,(radius,)*4,mode='reflect'),kernel,groups=3)
        return self.model((image-mean)/std)


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--transform',required=True)
    args,remaining=parser.parse_known_args();spec=json.loads(args.transform)
    if len(spec)!=1:raise ValueError('Each extra inference factor must contain exactly one setting')
    plan_index=remaining.index('--plan')+1
    plan=json.loads(open(remaining[plan_index]).read())
    assert len(plan)==1 and plan[0].get('forward_transform')==spec
    original_loader=runner.load_model
    def load(path,device):
        model=original_loader(path,device)
        if 'decoder_interpolation' in spec:
            for block in model.decoder.blocks.values():block.interpolation_mode=spec['decoder_interpolation']
            return model
        return ForwardAdapter(model,spec).to(device).eval()
    runner.load_model=load
    sys.argv=[sys.argv[0],*remaining]
    runner.main()


if __name__=='__main__':main()
