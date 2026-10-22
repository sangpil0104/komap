"""Rescore PNGs, verify state/storage claims, compare repeated wall times."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import hashlib, importlib.util, json, re, statistics, tarfile
from pathlib import Path
ROOT = Path('/workspace/model349_compress/base12_resident_packed_20261008')
OLD = Path('/workspace/model349_compress/audit_20261008/runs/cpu2_fp16_12')
GT = Path('/workspace/komap_model1/data/valid/masks')
def read(p): return json.loads(p.read_text())
def write(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')
def module(name, p):
    spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def main():
    done=read(ROOT/'completion.json'); assert done['base_and_weights_unchanged']
    scorer=module('scores','/workspace/model349_compress/audit_20261008/analyze_runs.py')
    helper=module('agreement','/workspace/model349_compress/base12_followup_20261008/finalize.py')
    references={'base12':OLD,'lite15':OLD.parent/'cpu2_lite15','original349':OLD.parent/'historical349'}
    scores={k:scorer.score(p,GT) for k,p in references.items()}
    runs={}; restore_audits={}
    for p in sorted((ROOT/'runs').iterdir()):
        if not (p/'metrics.json').exists(): continue
        val=scorer.score(p,GT)
        times={c:read(p/'probabilities'/c/'complete.json')['elapsed_seconds'] for c in ['326','329','base','336','269']}
        runs[p.name]={'miou_percent':100*val['miou'],'class_iou_percent':[100*x for x in val['classes']], 'completion':read(p/'completion.json'),'component_seconds':times,**{'vs_'+k:scorer.pair(ref,p,scores[k],val) for k,ref in references.items()}}
        log=(p/'logs/base.log').read_text()
        restores=re.findall(r'RESIDENT source=(\S+) restore=(\d+) cpu_sources=(\d+)',log)
        if restores:
            assert [int(x[1]) for x in restores]==list(range(1,21))
            chosen=re.findall(r'DONE base \S+ source=(\S+)',log)
            assert len(chosen)==20 and [Path(x[0]).parent.name for x in restores]==chosen
            restore_audits[p.name]={'restore_before_every_image':True,'restores':len(restores),'cpu_sources':int(restores[-1][2]),'sources':[x[0] for x in restores]}
    frames=[json.loads(x) for x in (ROOT/'monitor.jsonl').read_text().splitlines()]
    uuid=next(line.split(',')[1].strip() for line in frames[0]['gpu'].splitlines() if line.startswith('7,'))
    interference=[x['time'] for x in frames if sum(uuid in line for line in x['processes'].splitlines())>1]
    timings=None; selected=done['time_candidate']
    if selected:
        rows=read(ROOT/'timing_runs.json')
        values={k:[r['completion']['elapsed_seconds'] for r in rows if r['candidate']==k] for k in ['control12',selected]}
        assert all(len(v)==3 for v in values.values())
        med={k:statistics.median(v) for k,v in values.items()}
        reduction=100*(1-med[selected]/med['control12'])
        quality=all(runs[r['run']]['vs_base12']['quality_pass'] and runs[r['run']]['vs_lite15']['quality_pass'] for r in rows)
        paired=[100*(1-values[selected][i]/values['control12'][i]) for i in range(3)]
        storage=read(ROOT/'bundles'/selected/'checkpoint_manifest.json')
        timings={'candidate':selected,'seconds':values,'median_seconds':med,'median_reduction_percent':reduction,'round_paired_reduction_percent':paired,'all_repeats_quality_pass':quality,'ten_percent_gate_pass':reduction>=10 and quality and not interference,'checkpoint_count':storage['count'],'checkpoint_bytes':storage['bytes'],'bytes_reduction_percent':100*(1-storage['bytes']/683320900),'scope':'shared P100 GPU7 cross-ordered full valid20; all startup and fusion included'}
    tests={}
    for key in done['test_candidates']:
        p=ROOT/'runs'/(key+'_test10'); assert read(p/'completion.json')['mean_image_miou'] is None
        tests[key]={'vs_base12':helper.agreement(Path('/workspace/model349_compress/generalization_20261008/runs/cpu2_fp16_12_test10'),p),'completion':read(p/'completion.json')}
    raw=read(ROOT/'raw_comparison.json')
    storage_verification={p.parent.name:read(p) for p in (ROOT/'bundles').glob('*/lossless_storage_verification.json')}
    canonical=read(ROOT/'canonical_lossless_storage_verification.json')
    assert canonical['original_base_bytes']==683320900 and canonical['packed_bytes']==read(ROOT/'bundles/packed_fast12/checkpoint_manifest.json')['bytes']
    assert len(canonical['files'])==12 and all(r['zip_entries_exact'] and r['all_checkpoint_tensors_exact'] for r in canonical['files'])
    result={'runs':runs,'timings':timings,'test_mask_agreement':tests,'resident_reset_probe':read(ROOT/'probe_result.json'),'resident_restore_audits':restore_audits,'lossless_storage_verification':storage_verification,'canonical_lossless_storage_verification':canonical,'gpu7_multiple_process_frames':interference,'png_rescore_verified':True,'base_and_weights_unchanged':True,'independent_generalization_verified':False}
    write(ROOT/'comparison.json',result)
    lines=['# 12개 base 추가 탐색 결과','','고정 후보와 사전 이중 품질 조건을 적용했다. 같은 valid20 탐색이며 독립 일반화 검증이 아니다. 원본 base를 변경하지 않았다.','','| 후보 | 가중치 / 바이트 | valid image mIoU | base 평균 변화 | 최악 이미지 변화 | base / lite15 gate |','|---|---:|---:|---:|---:|---|']
    for key,item in raw.items():
        row=runs[key+'_valid20']; b=row['vs_base12']; ck=item['checkpoints']
        lines.append(f"| {key} | {ck['count']} / {ck['bytes']:,} | {row['miou_percent']:.7f}% | {b['delta_pp']:+.7f}%p | {b['worst_image_delta_pp']:+.7f}%p ({b['worst_image']}) | {b['quality_pass']} / {row['vs_lite15']['quality_pass']} |")
    lines+=['','공식 평가는 각 이미지 4 IoU 평균 후 20장 평균, absent/empty=0. 원본 영상 재추론과 PNG/GT 독립 재채점 일치를 확인했다. 모든 클래스·이미지별 손실과 기존 original349 비교는 comparison.json에 포함한다. raw 단회 시간은 속도 개선 근거가 아니다.','','## 상태 복원 및 저장 확인','',f"- 실제 GPU 복원 probe: {result['resident_reset_probe']}.",f'- full valid 복원 감사: {len(restore_audits)}개 실행에서 매 이미지 모든 source 복원 로그 확인. 원본 CPU state cache는 이미지 적응 tensor와 분리한다.']
    for key,rows in storage_verification.items():
        lines.append(f"- {key}: {len(rows)}파일의 모든 ZIP entry 해시 및 실제 state_dict tensor의 값/shape/dtype가 압축 전과 정확히 같다. 압축은 tensor 양자화나 연산 정밀도 변경이 아니다.")
    lines += [f"- 최종 기준은 별도 원본 base 직접 대조: {canonical['original_base_bytes']:,}→{canonical['packed_bytes']:,}바이트, 원본 12파일의 모든 ZIP entry 및 checkpoint 전체 tensor 정확히 동일. canonical_lossless_storage_verification.json을 확인한다.", '- 최초 압축 검사에서 B4/HRNet의 다른 checkpoint 키 구조를 처리하지 못해 중단됐다. generic tensor 재귀 검사로 고치고 완료된 추론을 다시 실행하지 않은 채 재개했다. 초기 코드와 failure 기록을 보관한다. 재개 과정의 두 파일 old_bytes는 이미 부분 압축된 파일 크기이므로 원본 기준 용량 계산에 쓰지 않는다. 원래 base 문서의 683,320,900바이트와 모든 원본 해시는 정확했다. 원본 파일 직접 대조 결과를 최종 기준으로 사용한다.']
    lines+=['- soup11은 seed456/pairwise 실수 가중치 0.5:0.5 평균 후 FP16 저장, 정수 buffer 최대값이다. 합성 provenance를 따로 기록했다. 새로운 학습 epoch/독립 일반화 성능은 없다.','- HRNet은 FP16 autocast + tile batch8의 결합 변경이며 AMP 단독 효과로 주장하지 않는다. 실제 convolution dtype은 logs/336.log에 기록했다.','','## 반복 시간','']
    if timings:
        lines += [f"선택 후보: {selected} (사전 우선순위).",f"- 기준 3회: {values['control12']}초 / 중앙값 {med['control12']:.2f}초.",f"- 후보 3회: {values[selected]}초 / 중앙값 {med[selected]:.2f}초.",f"- 중앙값 단축 {reduction:+.3f}%, 교차 회차별 {paired}%. 모든 반복 품질 {quality}, 10% gate {timings['ten_percent_gate_pass']}.",f"- 저장량 감소 {timings['bytes_reduction_percent']:.3f}%, 가중치 {storage['count']}개."]
    else: lines+=['모든 후보가 이중 품질 조건을 넘어서 반복 시간/test를 시작하지 않았다.']
    lines+=[f'- GPU7 복수 compute process 감지 {len(interference)} frames. CPU/GPU 부하 telemetry/monitor를 보관했다. 공유 서버에서의 측정이며 다른 환경 재현 속도를 보증하지 않는다.','','## test10','']
    for key,item in tests.items():
        r=item['vs_base12']; lines.append(f"- {key}: base와 {100*r['agreement']:.7f}% 마스크 일치, {r['changed_pixels']:,}/{r['pixels']:,}픽셀 변화. 원본 test10 추론 {item['completion']['elapsed_seconds']:.2f}초 단회. 정답이 없어 test IoU/정확도가 아니다.")
    lines+=['','## 제한 및 사용 상태','','100장뿐이며 동일 valid가 반복 후보 선택에 노출되었다. base의 specimen 중복·fold 선택·학습 이력/OOF 한계는 유지한다. 신규 데이터에 대한 일반화 개선을 주장하지 않는다. 기존 base와 가중치를 자동 교체하지 않고 실험 코드·설정·출력만 별도 보관했다. 이 판의 성능 수치가 높아도 교사의 학습 오류를 해소한 증류/새 학습 결과가 아니다.','']
    (ROOT/'report.md').write_text('\n'.join(lines))
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in ['.pt','.pyc'] and 'probabilities' not in p.parts and '__pycache__' not in p.parts and p.name not in ['gpu7.lock','results_archive.tar.gz','archive_receipt.json']]
    archive=ROOT/'results_archive.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        for p in sorted(files): stream.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    write(ROOT/'archive_receipt.json',{'files':len(files),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    print(json.dumps({'candidates':{k:runs[k+'_valid20']['miou_percent'] for k in raw},'timings':timings,'tests':{k:v['vs_base12']['agreement'] for k,v in tests.items()}},indent=2))
if __name__=='__main__': main()
