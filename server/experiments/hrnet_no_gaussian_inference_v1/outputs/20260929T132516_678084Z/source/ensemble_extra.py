"""Ablate averages of cached predictions from the SAME frozen checkpoint."""
import argparse
import json
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from ensemble_rules import combine
from tqdm import tqdm

from common import ROOT,DEADLINE,require_environment,data_rows,reserve_result,save_json,score_image,finalize,update_comparison


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
    require_environment();entries=json.loads((ROOT/'comparison.json').read_text());lookup={e['arm']['id']:e for e in entries}
    arms=json.loads(args.plan.read_text());run=ROOT/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    save_json(run/'plan.json',arms);print('RUN:',run,flush=True)
    (run/'source').mkdir()
    for source in ROOT.glob('*.py'):(run/'source'/source.name).write_bytes(source.read_bytes())
    rows=data_rows()
    for arm in tqdm(arms,desc='Same-model inference ensemble',unit='condition'):
        if time.time()>=DEADLINE:break
        weights=np.asarray(arm['weights'],np.float32);assert len(weights)==len(arm['members']) and (weights>=0).all() and weights.sum()>0
        weights=weights/weights.sum();members=[lookup[name] for name in arm['members']]
        assert all(m['checkpoint']==members[0]['checkpoint'] for m in members)
        d=run/'arms'/arm['id'];d.mkdir(parents=True);(d/'probs').mkdir();result=reserve_result();save_json(d/'result.json',dict(result=str(result)))
        began=time.time()
        for row in rows:
            if time.time()>=DEADLINE:raise TimeoutError('Deadline reached')
            values=[np.load(Path(m['cache'])/f"{row['image_id']}.npy") for m in members]
            q=combine(values,arm)
            np.save(d/'probs'/f"{row['image_id']}.npy",q);score_image(q,row,result)
        s=finalize(result,arm,time.time()-began,dict(cache=str(d/'probs')))
        save_json(d/'complete.json',dict(result=str(result),image_miou=s['image_miou']))
        print(arm['id'],f"{100*s['image_miou']:.6f}%",flush=True);update_comparison()


if __name__=='__main__':main()
