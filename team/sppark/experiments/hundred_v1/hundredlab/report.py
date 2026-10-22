from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from alphase.utils import sha256, write_json
from hundredlab.workspace import ROOT, PROJECT, BASE, output_path, read_json, run_dir, config
from hundredlab.catalog import catalog, REFERENCES
from hundredlab.evaluation import metric_summary


def baseline(mode):
    sub='evaluation_d4_review' if mode=='d4' else 'evaluation'
    return pd.read_csv(BASE/'run'/sub/'per_image_metrics.csv').set_index('image_id')


def completed_rows(smoke=False):
    rows=[]
    for name,item in catalog().items():
        run=run_dir(name,smoke)
        status=read_json(run/'status.json') if (run/'status.json').is_file() else {}
        for mode in ('plain','d4'):
            directory=output_path('smoke_evaluations' if smoke else 'evaluations',name,mode)
            if not (directory/'provenance.json').is_file():continue
            provenance=read_json(directory/'provenance.json')
            if provenance['request']['smoke_test']!=smoke:raise ValueError('Smoke/real mismatch')
            for filename,digest in provenance['artifacts'].items():
                if sha256(directory/filename)!=digest:raise ValueError(f'Evaluation changed: {directory/filename}')
            frame=pd.read_csv(directory/'per_image_metrics.csv').set_index('image_id')
            ref=baseline(mode)
            if set(frame.index)!=set(ref.index) or len(frame)!=20:raise ValueError('Validation IDs mismatch')
            frame=frame.loc[ref.index];delta=frame.mIoU_exclude-ref.mIoU_exclude
            cms=np.stack([json.loads(v) for v in frame.confusion_matrix])
            common=(cms.sum(2)>0).all(1)
            initial=read_json(run/'initial_score.json')['image_mIoU']
            history=[json.loads(v) for v in (run/'metrics.jsonl').read_text().splitlines()]
            trained_best=max(v['validation_miou'] for v in history if 'validation_miou' in v)
            row=dict(experiment=name,number=item['number'],group=item['group'],family=item.get('family','local'),
                description=item['description'],parent=item.get('parent','baseline') or 'scratch',inference=mode,
                **metric_summary(frame),delta_baseline_pp=float(delta.mean()),improved_images=int((delta>1e-8).sum()),
                common19_delta_pp=float(delta.to_numpy()[common].mean()),
                initial_plain_mIoU=initial,trained_best_plain_mIoU=trained_best,
                delta_initial_plain_pp=trained_best-initial,
                safe_checkpoint_uses_initial=trained_best<=initial,
                updates=status.get('update',0),minutes=status.get('elapsed_seconds',0)/60,
                complete=status.get('state')=='completed',best_trained_update=provenance['request']['optimizer_step'],
                trained_checkpoint=str(run/'best_trained.pt'),safe_checkpoint=str(run/'best.pt'),
                source=str(directory/'per_image_metrics.csv'))
            for c in range(4):
                row[f'iou_{c}']=float(frame[f'IoU_{c}'].mean())
                row[f'iou_{c}_image_count']=int(frame[f'IoU_{c}'].notna().sum())
            rows.append(row)
    return rows


def render_report(rows,failures=None,smoke=False):
    failures=failures or {};entries=catalog()
    done={r['experiment'] for r in rows if r['inference']=='d4' and r['complete']}
    all_done=len(done)==100 and all(any(r['experiment']==n and r['inference']=='plain' for r in rows) for n in done)
    lines=['# model_results_0920 — 100개 모델 실험 결과', '',
        f"상태: {'SMOKE 기능 점검 — 본 실험 아님' if smoke else ('100/100 완료' if all_done else f'진행 중 / 일부 완료: {len(done)}/100')}", '',
        '주 지표는 공식 검증 20장의 이미지별 평균 mIoU입니다. Global mIoU는 별도 보조 지표입니다. 값은 %, 변화는 pp입니다.', '',
        '구성: 추천 1개 + 기존/동료 결과 기반 19개 + 별도 탐색 80개. 100개의 서로 다른 실험 설정이며 모두 서로 다른 원 논문 모델 100종이라는 뜻은 아닙니다.', '',
        '각 행은 추가 학습 중 최고 plain 검증 점수로 선택한 best_trained.pt를 plain/D4로 평가한 결과입니다. best.pt는 학습 전 update 0까지 포함한 최고 plain 체크포인트입니다. 부모 가중치가 선택된 행은 별도로 표시합니다.', '']
    if rows:
        d4=sorted((r for r in rows if r['inference']=='d4'),key=lambda r:r['image_mIoU'],reverse=True)
        base=metric_summary(baseline('d4'))
        lines += [f"기존 기준 CNN+D4: 이미지 평균 {base['image_mIoU']:.4f}, Global {base['global_mIoU']:.4f}.",
            '이전 seed2028+D4의 관측 최고: 이미지 평균 78.7858, Global 81.5082. 빈 클래스 처리와 A71 민감도를 함께 확인해야 합니다.', '']
        if d4:
            best=d4[0]
            lines += [f"이번 관측 최고 후보: **{best['experiment']}** — 이미지 평균 **{best['image_mIoU']:.4f}**, Global **{best['global_mIoU']:.4f}**, 기준 대비 **{best['delta_baseline_pp']:+.4f} pp**.",
                f"이전 이미지 평균 최고 78.7858 대비 {best['image_mIoU']-78.78576023358842:+.4f} pp. 관측 검증 최고이며 독립 테스트 성능을 뜻하지 않습니다.", '']
            lines += ['## D4 순위', '', '| 순위 | 실험 | 그룹 | 이미지 mIoU | Global mIoU | 기준 Δpp | 공통19장 Δpp | 개선 장수 | 초기 가중치 유지 | 학습 분 |',
                      '|---:|---|---|---:|---:|---:|---:|---:|---|---:|']
            for i,r in enumerate(d4,1):
                lines.append(f"| {i} | {r['experiment']} | {r['group']} | {r['image_mIoU']:.4f} | {r['global_mIoU']:.4f} | {r['delta_baseline_pp']:+.4f} | {r['common19_delta_pp']:+.4f} | {r['improved_images']}/20 | {'예' if r['safe_checkpoint_uses_initial'] else '아니오'} | {r['minutes']:.2f} |")
            lines += ['', '## 상위 후보의 클래스별 IoU', '', '| 실험 | 초정 Si | Al3Ni | 공정 Si | Al |', '|---|---:|---:|---:|---:|']
            for r in d4[:10]:lines.append('| '+r['experiment']+' | '+' | '.join(f"{r[f'iou_{c}']:.4f}" for c in range(4))+' |')
            lines += ['', '## 평가 해석', '',
                f"- 기준 모델보다 높은 D4 후보: {sum(r['delta_baseline_pp']>0 for r in d4)}/{len(d4)}개.",
                f"- 초기 가중치를 plain 기준으로 넘지 못한 후보: {sum(r['safe_checkpoint_uses_initial'] for r in d4)}/{len(d4)}개.",
                f"- 완료 모델의 학습 시간 합: {sum(r['minutes'] for r in d4):.2f}분. 프로세스 시작·최종 평가 시간 등은 별도입니다.",
                '- warm-start와 scratch는 시작 가중치 및 학습 예산이 다릅니다. 구조 효과만의 공정한 비교로 해석하지 마세요.',
                '- 100개 후보와 여러 체크포인트를 같은 validation에서 선택했습니다. 최고값은 개발 점수이며 그룹 CV/새 검증셋과 반복 seed로 확인해야 합니다.',
                '- A71의 빈 클래스 처리에 따른 점수 변화를 보기 위해 동일 19장 변화와 클래스별 유효 이미지 수도 CSV에 기록합니다.',
                '- 공식 train/validation 사이 이미지 ID 중복은 없지만 일부 A-ID 그룹은 공유됩니다. 다른 fold는 해당 fold의 새 부모 모델부터 학습해야 합니다.',
                '- 동료 문서의 79.48은 Global mIoU, 75.83은 이미지 평균입니다. 동일 평가 CSV·클래스 매핑이 없어 직접적인 통제 비교는 아닙니다.',
                '- test 10장에는 정답이 없으므로 테스트 정확도는 보고하지 않습니다. 기존 모델은 자동 교체하지 않습니다.', '']
    lines += ['## 100개 설정 및 실행 상태', '', '| 번호 | 실험 | 상태 | 초기화 | 설명 |', '|---:|---|---|---|---|']
    for name,item in entries.items():
        state='완료' if name in done else ('실패' if name in failures else '미완료')
        lines.append(f"| {item['number']} | {name} | {state} | {item.get('parent','baseline') or 'scratch'} | {item['description']} |")
    if failures:
        lines += ['', '## 실패와 재실행', '', '같은 전체 실행 명령을 다시 실행하면 완료 결과는 검사 후 재사용하고 미완료 모델을 재개합니다.', '']
        lines += [f'- `{n}`: {str(error).replace(chr(10)," ")[:500]}' for n,error in failures.items()]
    lines += ['', '## 재현 자료', '',
        '- 실험 코드: `experiments/hundred_v1/`; 원본 모델·소스·결과 보존.',
        '- 설정: `experiments/hundred_v1/configs/resolved/` 및 각 run의 `resolved_config.yaml`.',
        '- 세부 결과: `experiments/hundred_v1/outputs/reports/results.csv` (plain/D4, 클래스 점수, 체크포인트 경로).',
        '- 로그/체크포인트: `experiments/hundred_v1/outputs/runs/<실험>/`.',
        '- 출처의 아이디어를 경량 변형한 자체 구현입니다. 공식 사전학습 백본이나 논문 전체 모델을 재현한 것은 아닙니다.', '', '## 참고 문헌', '']
    lines += [f'- [{key}]({url})' for key,url in REFERENCES.items()]
    return '\n'.join(lines)+'\n',all_done


def report(failures=None,smoke=False,publish=False):
    rows=completed_rows(smoke)
    folder=output_path('smoke_reports' if smoke else 'reports');folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/'results.csv',index=False)
    text,all_done=render_report(rows,failures,smoke)
    (folder/'model_results_0920.md').write_text(text)
    if publish and all_done and not smoke:
        target=PROJECT/'model_results_0920.md'
        if target.is_symlink():raise ValueError('Report target cannot be a symlink')
        if target.exists() and target.read_text()!=text:
            backups=output_path('report_history');backups.mkdir(parents=True,exist_ok=True)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            shutil.copy2(target,backups/f'model_results_0920_before_{stamp}.md')
        temporary=PROJECT/'.model_results_0920.hundred.tmp'
        temporary.write_text(text);temporary.replace(target)
        print(f'100개 완료 보고서: {target}',flush=True)
    write_json(folder/'completion.json',dict(completed=all_done,smoke=smoke,rows=len(rows)))
    return rows,all_done
