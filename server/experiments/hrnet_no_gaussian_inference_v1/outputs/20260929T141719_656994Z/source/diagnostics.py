"""Image-level diagnostics and inference pixel-work proxy; no pooled IoU."""
import json
import math

import numpy as np
from PIL import Image

from common import ROOT, data_rows, save_json
from inference import starts, VIEWS
from run_adaptive import realize


def main():
    entries=json.loads((ROOT/'comparison.json').read_text())
    lookup={e['arm']['id']:e for e in entries}
    base=lookup['A00_control']
    base_rows={r['image_id']:r for r in base['rows']}
    sizes={r['image_id']:Image.open(r['image']).size for r in data_rows()}
    costs={}
    def cost(name):
        if name in costs:return costs[name]
        arm=lookup[name]['arm']
        if 'members' in arm:
            # Distinct dependency nodes are counted once even if a DAG reuses them.
            leaves={}
            for member in arm['members']:leaves.update(cost(member))
        elif arm.get('change',{}).get('kind'):
            leaves=cost(arm['contrast'])
        elif 'recipe' in arm:
            leaves=cost(arm['contrast'])
        else:
            cfg=arm['config'];pixels=0
            for width,height in sizes.values():
                c=realize(cfg,width,arm['adaptive_scale']) if arm.get('adaptive_scale') else cfg
                if 'patch' in c:
                    native=round(c['patch']/c['scale']);stride=native-c['overlap']
                    width+=2*c['halo'];height+=2*c['halo']
                    count=len(starts(max(width,native),native,stride))*len(starts(max(height,native),native,stride))
                    pixels+=count*c['patch']**2*len(VIEWS[c['tta']])
                else:
                    width+=2*c.get('halo',0);height+=2*c.get('halo',0)
                    h=math.ceil((round(height*c['scale'])+2*c.get('model_halo',0))/32)*32
                    w=math.ceil((round(width*c['scale'])+2*c.get('model_halo',0))/32)*32
                    pixels+=h*w*len(VIEWS[c['tta']])
            leaves={name:pixels}
        costs[name]=leaves
        return leaves
    base_cost=sum(cost('A00_control').values())
    best=max(entries,key=lambda e:e['image_miou'])
    rows=[]
    for row in best['rows']:
        name=row['image_id'];b=base_rows[name]
        rows.append(dict(image_id=name,size=sizes[name],baseline=b['image_miou'],best=row['image_miou'],
                         delta_pp=100*(row['image_miou']-b['image_miou']),
                         boundary_delta_pp=100*(row['boundary_error']['1']-b['boundary_error']['1'])))
    ranking=[]
    for e in sorted(entries,key=lambda e:e['image_miou'],reverse=True):
        name=e['arm']['id']
        ratio=sum(cost(name).values())/base_cost
        ranking.append(dict(id=name,image_miou=e['image_miou'],pixel_work_ratio=ratio))
    small=[r for r in rows if r['size'][0]<=800]
    large=[r for r in rows if r['size'][0]>800]
    result=dict(best=best['arm']['id'],rows=rows,work=ranking,
                small_images=dict(count=len(small),delta_pp=float(np.mean([r['delta_pp'] for r in small]))),
                large_images=dict(count=len(large),delta_pp=float(np.mean([r['delta_pp'] for r in large]))))
    save_json(ROOT/'diagnostics.json',result)
    lines=['# 이미지별 진단과 연산량 비교','',f"현재 최고 조건: `{best['arm']['id']}` ({best['image_miou']*100:.6f}%).",'',
           '모든 수치는 같은 valid20에서 탐색한 결과이며, 새로운 데이터에서의 개선을 보장하지 않습니다.',
           '경계 오류는 정답 상 사이 경계 주변에서 틀린 픽셀 비율입니다. 음수 변화량이 개선입니다.',
           '입력 픽셀 처리량은 타일 수 × 타일 면적 × TTA 수의 합입니다. 실제 시간이나 FLOPs 측정값은 아닙니다.',
           '중첩된 ensemble에서 동일한 추론 결과는 한 번만 계산한 것으로 셉니다.','',
           f"원본 너비 ≤800: {len(small)}장, 평균 변화 {result['small_images']['delta_pp']:+.4f}%p.",
           f"원본 너비 >800: {len(large)}장, 평균 변화 {result['large_images']['delta_pp']:+.4f}%p.",'',
           '| 이미지 | 원본 크기 | 기본 image mIoU | 최고 조건 | 변화 (%p) | 경계 오류 변화 (%p) |',
           '|---|---|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['delta_pp']):
        lines.append(f"| {r['image_id']} | {r['size'][0]}×{r['size'][1]} | {r['baseline']*100:.4f}% | {r['best']*100:.4f}% | {r['delta_pp']:+.4f} | {r['boundary_delta_pp']:+.4f} |")
    lines+=['','| 조건 | image mIoU | 입력 픽셀 처리량 / 기본 |','|---|---:|---:|']
    for r in ranking[:30]:lines.append(f"| {r['id']} | {r['image_miou']*100:.6f}% | {r['pixel_work_ratio']:.3f}× |")
    (ROOT/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print(best['arm']['id'],result['small_images'],result['large_images'])


if __name__=='__main__':main()
