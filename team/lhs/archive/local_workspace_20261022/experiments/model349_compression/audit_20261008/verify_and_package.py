"""After timing: structural FP16 audit and independent portable candidate copy."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import torch
ROOT=Path('/workspace/model349_compress/audit_20261008'); BASE=ROOT.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
assert (ROOT/'batched_complete.json').is_file() and (ROOT/'hramp_complete.json').is_file(), 'Finish all measurements first'
spec=importlib.util.spec_from_file_location('fp16check',BASE/'verify_fp16_storage.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_num_threads(2)
results={}
for source,dest in [(BASE/'model349_lite14_ema_pruned',BASE/'lite14_fp16_storage'),(BASE/'model349_lite_rep_free_15weights_v2',ROOT/'bundles/fp16_15_batched')]:
    stats={'max_abs_error':0.0,'sum_abs_error':0.0,'float_elements':0,'changed_elements':0}
    original=sorted((source/'weights').rglob('*.pt')); converted=sorted((dest/'weights').rglob('*.pt'))
    assert len(original)==len(converted) in (14,15)
    report=[]
    for p in original:
        target=dest/p.relative_to(source)
        mod.compare(torch.load(p,map_location='cpu',weights_only=False),torch.load(target,map_location='cpu',weights_only=False),stats)
        report.append({'path':str(p.relative_to(source)),'source_sha256':sha(p),'converted_sha256':sha(target)})
    stats['mean_abs_error']=stats.pop('sum_abs_error')/stats['float_elements']; stats['checkpoints']=len(original)
    stats['source_bytes']=sum(p.stat().st_size for p in original); stats['converted_bytes']=sum(p.stat().st_size for p in converted)
    stats['files']=report; results[str(dest)]=stats
    print('Verified',dest,stats['checkpoints'],flush=True)
write(ROOT/'fp16_structural_audit.json',results)
# The source SHA in conversion reports must still match the untouched original files.
for dest in [BASE/'lite14_fp16_storage',ROOT/'bundles/fp16_15_batched']:
    rows=json.loads((dest/'fp16_storage_report.json').read_text())
    source=BASE/('model349_lite14_ema_pruned' if dest.name=='lite14_fp16_storage' else 'model349_lite_rep_free_15weights_v2')
    for row in rows:
        assert row['source_sha256']==sha(source/row['path'])
        assert row['output_sha256']==sha(dest/row['path'])
import cv2, numpy, scipy, PIL, yaml, segmentation_models_pytorch, timm
versions={'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'opencv':cv2.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'pillow':PIL.__version__,'pyyaml':yaml.__version__,'segmentation_models_pytorch':segmentation_models_pytorch.__version__,'timm':timm.__version__}
write(ROOT/'validation_runtime_versions.json',versions)
shutil.copy2(ROOT/'validation_runtime_versions.json',ROOT/'bundles/fp16_15_batched/validation_runtime_versions.json')
summary=json.loads((ROOT/'batched_summary.json').read_text()); selected='fp16_15_batched'; selected_report='batched_summary.json'; selected_test='batched_test_parity.json'
if (ROOT/'hramp_summary.json').exists():
    extra=json.loads((ROOT/'hramp_summary.json').read_text())
    if extra.get('exploratory_adoption_pass'):
        summary=extra; selected='fp16_15_batched_hramp'; selected_report='hramp_summary.json'; selected_test='hramp_test_parity.json'
write(ROOT/'selected_variant.json',{'variant':selected if summary.get('exploratory_adoption_pass') else None,'report':selected_report,'test_parity':selected_test})
if summary.get('exploratory_adoption_pass'):
    source=ROOT/'bundles'/selected; dest=ROOT/('model349_lite15_fp16_d4batch4_hramp_candidate' if selected.endswith('hramp') else 'model349_lite15_fp16_d4batch4_candidate')
    assert not dest.exists(); shutil.copytree(source,dest)
    shutil.copy2(ROOT/'validation_runtime_versions.json',dest/'validation_runtime_versions.json')
    (dest/'provenance').mkdir(exist_ok=True)
    old_manifest=dest/'provenance/weights_manifest.json'
    if old_manifest.exists():
        old_manifest.rename(dest/'provenance/source_fp32_weights_manifest.json')
    write(dest/'provenance/weights_manifest.json',json.loads((source/'checkpoint_manifest.json').read_text())['checkpoints'])
    write(dest/'evaluation_provenance.json',{'validation':'Repeated raw-image valid20; four-class image mIoU; empty=0','comparison_report':str(ROOT/'comparison.json'),'timing_report':str(ROOT/selected_report),'test_parity':str(ROOT/selected_test),'independent_generalization_claimed':False,'source_release_overwritten':False})
    (dest/'README.md').write_text('''# Model349 lite15 FP16 저장 + D4 view batch4 검증 후보

15개 원래 lite 모델(EMA 포함)을 모두 유지한다. 저장 체크포인트는 FP16, 로드된 module은 FP32다. 기존 AMP 계산 경로를 유지하며 고해상도 768 D4 view를 4개씩 묶는다. TENT 5-step·raw/style·HRNet·269·B4 모든 fold·혼합 계수·후처리는 유지한다. OpenCV와 PyTorch는 각각 2 threads다.

평가·시간·test mask parity는 evaluation_provenance.json과 원 실험 comparison.json, batched_summary.json을 참조한다. 같은 valid20을 반복 사용한 재현 최적화이며 독립 일반화나 test mIoU를 주장하지 않는다. 기존 릴리스는 덮어쓰지 않았다.

CUDA GPU 점유를 확인한 뒤 새 output 경로로 infer_lite.py를 실행한다. 기존 output 폴더는 허용하지 않는다. 전체 추론 방법은 감사 README.md에 있다. 현재 검증 환경 버전은 validation_runtime_versions.json에 기록했다.
''')
    if selected.endswith('hramp'):
        (dest/'README.md').write_text((dest/'README.md').read_text()+'\n추가 변경: HRNet forward에 CUDA FP16 autocast를 적용하고 tile batch를 2에서 8로 늘린다. 타일 확률 누적 순서는 유지한다. 이는 저장 정밀도와 별도의 실제 연산 정밀도 변경이다. 상세 평가와 반복 시간은 hramp_summary.json을 참조한다.\n')
    manifest={'files':[{'path':str(p.relative_to(dest)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(dest.rglob('*')) if p.is_file()]}
    write(dest/'bundle_manifest.json',manifest)
    print('Portable candidate:',dest,flush=True)
write(ROOT/'final_verification.json',{'structure_verified':True,'checked_original_14_and_15_weight_hashes_unchanged':True,'portable_candidate_created':bool(summary.get('exploratory_adoption_pass'))})
