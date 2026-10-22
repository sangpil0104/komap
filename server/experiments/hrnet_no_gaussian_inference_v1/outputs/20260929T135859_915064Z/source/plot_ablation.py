"""Export static figures from completed validation scores."""
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, save_json


def main():
    entries=json.loads((ROOT/'comparison.json').read_text())
    lookup={e['arm']['id']:e for e in entries};base=lookup['A00_control']
    best=max(entries,key=lambda e:e['image_miou'])
    out=ROOT/'analysis'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    families=[('G_native_',256,'Native tile side (px)','Fixed 2x magnification; relative overlap fixed'),
              ('H_resize_',512,'Magnification','Fixed native tile=256; overlap=170')]
    for axis,(prefix,control_x,label,title) in zip(axes,families):
        points=[(control_x,lookup['G00_batch1_control']['image_miou']*100)]
        points += [(int(e['arm']['id'].removeprefix(prefix)),100*e['image_miou']) for e in entries if e['arm']['id'].startswith(prefix)]
        points=sorted(points);x,y=np.array(points).T
        if prefix=='H_resize_':x=x/256
        axis.plot(x,y,'o-',color='#1864ab',lw=1.8)
        axis.axhline(100*lookup['G00_batch1_control']['image_miou'],color='#888888',ls='--',lw=1)
        axis.set(xlabel=label,ylabel='Mean image mIoU (%)',title=title);axis.grid(alpha=.2)
    fig.suptitle('Frozen HRNet: context and magnification are different factors',fontsize=12)
    fig.savefig(out/'geometry.png',dpi=180);fig.savefig(out/'geometry.svg');plt.close(fig)
    base_rows={r['image_id']:r for r in base['rows']}
    rows=sorted(best['rows'],key=lambda r:r['image_miou']-base_rows[r['image_id']]['image_miou'])
    labels=[r['image_id'] for r in rows];delta=np.array([100*(r['image_miou']-base_rows[r['image_id']]['image_miou']) for r in rows])
    fig,axis=plt.subplots(figsize=(9,7),layout='constrained')
    axis.barh(labels,delta,color=np.where(delta>=0,'#2b8a3e','#c92a2a'));axis.axvline(0,color='black',lw=.7)
    axis.set(xlabel='Image mIoU change vs A00 (percentage points)',title=best['arm']['id']+' — exploratory valid20 effects')
    axis.grid(axis='x',alpha=.2)
    fig.savefig(out/'per_image.png',dpi=180);fig.savefig(out/'per_image.svg');plt.close(fig)
    save_json(out/'source.json',dict(best=best['arm']['id'],scores={e['arm']['id']:e['image_miou'] for e in entries},figures=['geometry.svg','per_image.svg']))
    save_json(ROOT/'latest_figures.json',dict(directory=str(out),best=best['arm']['id']))
    print(out)


if __name__=='__main__':main()
