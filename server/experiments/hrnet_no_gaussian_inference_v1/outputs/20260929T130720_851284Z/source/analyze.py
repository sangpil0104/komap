"""Summarize paired image-level effects without computing global IoU."""
import json
from pathlib import Path

import numpy as np

from common import ROOT,save_json,update_comparison


def main():
    update_comparison()
    source=ROOT/'comparison.json'
    if not source.exists():return
    entries=json.loads(source.read_text());by_id={e['arm']['id']:e for e in entries}
    if 'A00_control' not in by_id:return
    base=by_id['A00_control'];base_rows={r['image_id']:r for r in base['rows']}
    rng=np.random.default_rng(32);bootstrap=rng.integers(0,20,(10000,20))
    ranked=[]
    for e in entries:
        rows=sorted(e['rows'],key=lambda r:r['image_id'])
        delta=np.array([r['image_miou']-base_rows[r['image_id']]['image_miou'] for r in rows])
        interval=np.quantile(delta[bootstrap].mean(1),[.025,.975])*100
        ranked.append(dict(id=e['arm']['id'],image_miou=e['image_miou'],delta_pp=delta.mean()*100,
                           improved_images=int((delta>1e-9).sum()),worse_images=int((delta< -1e-9).sum()),
                           class_iou=e['class_iou'],paired_bootstrap_95ci_pp=interval.tolist(),result=e['result'],
                           boundary_error_1px=float(np.mean([r['boundary_error']['1'] for r in rows])),
                           change=e['arm'].get('change'),contrast=e['arm'].get('contrast')))
    ranked.sort(key=lambda r:r['image_miou'],reverse=True)
    save_json(ROOT/'ranking.json',ranked)
    lines=['# Exploratory validation ranking','',
           'Paired bootstrap resamples the 20 validation images. Intervals are descriptive and do not correct for selection among many tested conditions.',
           'The same validation set guides this search; an independent test or fresh validation split is needed for a generalization claim.','',
           '| ID | image mIoU | Delta vs A00 (pp) | Better / worse images | Paired 95% interval (pp) | Boundary error 1px |','|---|---:|---:|---|---|---:|']
    for r in ranked:
        lo,hi=r['paired_bootstrap_95ci_pp'];lines.append(f"| {r['id']} | {r['image_miou']*100:.6f}% | {r['delta_pp']:+.6f} | {r['improved_images']} / {r['worse_images']} | [{lo:+.4f}, {hi:+.4f}] | {r['boundary_error_1px']*100:.4f}% |")
    (ROOT/'RANKING.md').write_text('\n'.join(lines)+'\n')
    print('Completed conditions:',len(entries))
    for row in ranked[:12]: print(row['id'],f"{100*row['image_miou']:.6f}%",f"{row['delta_pp']:+.6f}pp",row['change'])


if __name__=='__main__':main()
