"""Ablate image-size-based magnification; no image IDs or labels enter the rule."""
import argparse
import copy
import json
import sys
from pathlib import Path

import run as runner
from common import save_json


def realize(cfg,width,policy):
    c=cfg.copy()
    if policy['kind']=='minimum_width':scale=max(cfg['scale'],min(policy['max_scale'],policy['width']/width))
    elif policy['kind']=='target_width':scale=max(policy['min_scale'],min(policy['max_scale'],policy['width']/width))
    elif policy['kind']=='sqrt_width':scale=cfg['scale']*(policy['reference_width']/width)**.5;scale=max(cfg['scale'],min(policy['max_scale'],scale))
    elif policy['kind']=='constant':scale=policy['scale']
    else:raise ValueError(policy)
    native=round(cfg['patch']/cfg['scale']);new_native=round(cfg['patch']/scale)
    c.update(scale=scale,overlap=round(cfg['overlap']*new_native/native))
    if not (0<=c['overlap']<new_native):raise ValueError('Invalid derived overlap')
    return c


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--policy',required=True);args,remaining=p.parse_known_args();policy=json.loads(args.policy)
    plan=json.loads(Path(remaining[remaining.index('--plan')+1]).read_text());assert len(plan)==1 and plan[0]['adaptive_scale']==policy
    original=runner.predict_group
    def predict(model,image,arms,device,batch_size,description='inference'):
        modified=copy.deepcopy(arms)
        for a in modified:
            a['config']=realize(a['config'],image.shape[-1],policy)
            save_json(Path(a['_dir'])/'realized_configs'/f'{description}.json',dict(image_width=image.shape[-1],policy=policy,config=a['config']))
        return original(model,image,modified,device,batch_size,description=description)
    runner.predict_group=predict
    sys.argv=[sys.argv[0],*remaining];runner.main()


if __name__=='__main__':main()
