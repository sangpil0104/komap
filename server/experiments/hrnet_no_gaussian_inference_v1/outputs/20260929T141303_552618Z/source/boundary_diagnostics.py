"""Descriptive boundary/interior and small-component error analysis."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from common import ROOT, data_rows, save_json
from metrics import mask_ids, boundary_band, CLASSES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--id',required=True);args=parser.parse_args()
    entries={e['arm']['id']:e for e in json.loads((ROOT/'comparison.json').read_text())}
    base=entries['A00_control'];best=entries[args.id];records=[];components=[]
    bins=[(1,4),(5,16),(17,64),(65,256),(257,10**12)]
    for row in tqdm(data_rows(),desc='Boundary diagnostics',unit='image'):
        name=row['image_id'];target=mask_ids(np.asarray(Image.open(row['mask']).convert('RGB')))
        predictions=[mask_ids(np.asarray(Image.open(Path(e['result'])/'predictions'/f'{name}.png'))) for e in [base,best]]
        band=boundary_band(target,1)
        for region,where in [('boundary',band),('interior',~band)]:
            before,after=[p[where]==target[where] for p in predictions]
            records.append(dict(image_id=name,region=region,pixels=int(where.sum()),
                                baseline_error=float((~before).mean()),candidate_error=float((~after).mean()),
                                corrected=int((~before&after).sum()),regressed=int((before&~after).sum())))
        for c in range(4):
            count,labels,stats,_=cv2.connectedComponentsWithStats((target==c).astype(np.uint8),connectivity=8)
            area=stats[1:,cv2.CC_STAT_AREA]
            for low,high in bins:
                selected=np.flatnonzero((area>=low)&(area<=high))+1
                if not len(selected):continue
                where=np.isin(labels,selected);recall=[float((p[where]==c).mean()) for p in predictions]
                components.append(dict(image_id=name,class_name=CLASSES[c],min_area=low,max_area=high,
                                       components=len(selected),pixels=int(where.sum()),
                                       baseline_recall=recall[0],candidate_recall=recall[1]))
    output=dict(id=args.id,regions=records,components=components)
    save_json(ROOT/'checks'/f'boundary_{args.id}.json',output)
    lines=[f'# 경계 진단: {args.id}','',
           '기본 A00과 비교합니다. 아래 오류·재현율은 보조 진단이며 공식 image mIoU를 대체하지 않습니다.',
           '상 사이 정답 경계의 양쪽 픽셀을 찾은 뒤 1픽셀 확장한 띠를 경계 영역으로 사용합니다.',
           '모든 평균은 이미지별 비율을 먼저 계산합니다. 해당 크기의 정답 상이 없는 이미지는 그 재현율 평균에서 제외합니다.','',
           '| 영역 | 기본 오류율 | 후보 오류율 | 변화 (%p) |','|---|---:|---:|---:|']
    for region in ['boundary','interior']:
        subset=[r for r in records if r['region']==region]
        before=np.mean([r['baseline_error'] for r in subset]);after=np.mean([r['candidate_error'] for r in subset])
        lines.append(f'| {region} | {before*100:.4f}% | {after*100:.4f}% | {(after-before)*100:+.4f} |')
    lines+=['','정답의 8방향 연결 성분 크기별 픽셀 재현율:','',
            '| 상 | 연결 성분 면적 (원본 px) | 이미지 수 | 기본 재현율 | 후보 재현율 | 변화 (%p) |',
            '|---|---|---:|---:|---:|---:|']
    for c in CLASSES:
        for low,high in bins:
            subset=[r for r in components if r['class_name']==c and r['min_area']==low]
            if not subset:continue
            before=np.mean([r['baseline_recall'] for r in subset]);after=np.mean([r['candidate_recall'] for r in subset])
            label=f'{low}–{high}' if high<10**12 else f'≥{low}'
            lines.append(f'| {c} | {label} | {len(subset)} | {before*100:.4f}% | {after*100:.4f}% | {(after-before)*100:+.4f} |')
    (ROOT/f'BOUNDARY_{args.id}.md').write_text('\n'.join(lines)+'\n')
    print('Saved boundary diagnosis for',args.id)


if __name__=='__main__':main()
