"""Retrospective selection sensitivity, explicitly not independent validation."""
from collections import Counter
import json

import numpy as np

from common import ROOT, save_json


def main():
    entries=json.loads((ROOT/'comparison.json').read_text())
    entries=[e for e in entries if not e['arm']['id'].startswith(('REPRO','V00_','V01_','N00_','W00_'))]
    names=sorted(r['image_id'] for r in entries[0]['rows'])
    names_index={n:i for i,n in enumerate(names)}
    scores=np.zeros((len(names),len(entries)))
    for col,e in enumerate(entries):
        for r in e['rows']:scores[names_index[r['image_id']],col]=r['image_miou']
    baseline=next(i for i,e in enumerate(entries) if e['arm']['id']=='A00_control')
    groups={n.split('_')[0] for n in names};records=[]
    for group in sorted(groups):
        test=np.array([n.split('_')[0]==group for n in names]);train=~test
        chosen=int(scores[train].mean(0).argmax())
        for i in np.flatnonzero(test):records.append(dict(image_id=names[i],held_out_specimen=group,
                                  chosen=entries[chosen]['arm']['id'],score=float(scores[i,chosen]),baseline=float(scores[i,baseline])))
    mean=float(np.mean([r['score'] for r in records]));base=float(scores[:,baseline].mean())
    rng=np.random.default_rng(32);counts=rng.multinomial(20,np.full(20,1/20),size=10000)
    chosen=(counts@scores/20).argmax(1);frequency=Counter(entries[i]['arm']['id'] for i in chosen)
    result=dict(conditional_on_explored_candidate_set=True,independent_validation=False,
                note='Candidate designs already used all20 feedback. Group omission only diagnoses selection sensitivity and cannot undo that leakage.',
                num_candidates=len(entries),specimen_groups=len(groups),baseline=base,global_best=float(scores.mean(0).max()),
                omitted_specimen_selection_score=mean,records=records,bootstrap_winner_frequency=dict(frequency.most_common()))
    save_json(ROOT/'selection_sensitivity.json',result)
    lines=['# 선택 민감도 진단','',
           '**독립 검증 점수가 아닙니다.** 후보 설계 자체가 valid20의 피드백을 이용했으므로, 아래 사후 분석으로 그 영향을 제거할 수 없습니다.',
           '같은 A번호의 이미지를 묶어서 한 그룹씩 제외하고, 남은 이미지에서 최고 조건을 고른 뒤 제외한 이미지의 저장된 예측을 확인했습니다.',
           '새 가중치 학습이나 이미지 ID에 따른 배포 규칙은 만들지 않습니다.','',
           f"후보 {len(entries)}개 / A번호 그룹 {len(groups)}개.",
           f"기본 {base*100:.6f}%; 전체 이미지로 고른 최고 {result['global_best']*100:.6f}%; 그룹 제외 선택 후 합산 {mean*100:.6f}%.",'',
           '아래는 valid20 이미지를 10,000번 복원추출했을 때의 최고 조건 빈도입니다. 일반화 확률이 아닙니다.','',
           '| 조건 | 선택 빈도 |','|---|---:|']
    for name,n in frequency.most_common(15):lines.append(f'| {name} | {n/100:.2f}% |')
    (ROOT/'SELECTION_SENSITIVITY.md').write_text('\n'.join(lines)+'\n')
    print('Sensitivity only:',result['global_best']*100,mean*100,frequency.most_common(5))


if __name__=='__main__':main()
