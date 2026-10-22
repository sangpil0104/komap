"""Independently recompute image IoU from saved RGB predictions and masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import ROOT,require_environment,data_rows,sha,save_json
from metrics import PALETTE,mask_ids


def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--report',required=True);p.add_argument('--skip-verified',action='store_true');args=p.parse_args()
    require_environment();entries=json.loads((ROOT/'comparison.json').read_text())
    if args.ids:entries=[e for e in entries if e['arm']['id'] in args.ids]
    if args.skip_verified:
        verified=set()
        for previous in (ROOT/'checks').glob('audit_*.json'):
            for record in json.loads(previous.read_text()).get('conditions',[]):
                if record.get('score_verified') and record.get('panel_pixels_verified') and record.get('artifacts_verified'):
                    verified.add(record['id'])
        entries=[e for e in entries if e['arm']['id'] not in verified]
    report=ROOT/'checks'/args.report
    if report.exists():raise FileExistsError(report)
    records=[]
    for entry in tqdm(entries,desc='Independent result audit',unit='condition'):
        result=Path(entry['result']);manifest=json.loads((result/'complete.json').read_text())['files']
        for path,digest in manifest.items():assert sha(result/path)==digest,(result,path)
        scores=[]
        for row in data_rows():
            name=row['image_id'];pred=mask_ids(np.asarray(Image.open(result/'predictions'/f'{name}.png')))
            target=mask_ids(np.asarray(Image.open(row['mask']).convert('RGB')))
            prob=np.load(Path(entry['cache'])/f'{name}.npy',mmap_mode='r')
            assert prob.shape==(4,*pred.shape) and np.isfinite(prob).all() and prob.min()>=0
            np.testing.assert_allclose(prob.sum(0),1,atol=2e-5)
            np.testing.assert_array_equal(prob.argmax(0),pred)
            ious=[]
            for c in range(4):
                union=((pred==c)|(target==c)).sum();tp=((pred==c)&(target==c)).sum()
                ious.append(float(tp/union) if union else 0.)
            score=float(np.mean(ious));saved=json.loads((result/'per_image'/f'{name}.json').read_text())
            assert abs(score-saved['image_miou'])<1e-12;np.testing.assert_allclose(ious,saved['class_iou'],atol=1e-12)
            scores.append(score)
            panel=np.asarray(Image.open(result/'valid_mask_comparison'/f'{name}.png'))
            h,w=target.shape;gap=24;header=108;caption=44;y=header+caption;y2=header+h+caption+gap+caption
            image=np.asarray(Image.open(row['image']).convert('RGB'));err=np.full((h,w,3),255,np.uint8);wrong=pred!=target;err[wrong]=PALETTE[target[wrong]]
            np.testing.assert_array_equal(panel[y:y+h,:w],image)
            np.testing.assert_array_equal(panel[y:y+h,w+gap:w+gap+w],err)
            np.testing.assert_array_equal(panel[y2:y2+h,:w],PALETTE[pred])
            np.testing.assert_array_equal(panel[y2:y2+h,w+gap:w+gap+w],PALETTE[target])
        assert abs(np.mean(scores)-entry['image_miou'])<1e-12
        records.append(dict(id=entry['arm']['id'],images=20,score_verified=True,panel_pixels_verified=True,artifacts_verified=True))
    save_json(report,dict(preservation_verified=True,conditions=records))
    print('Verified',len(records),'conditions and',20*len(records),'masks/panels; original artifacts unchanged.')


if __name__=='__main__':main()
