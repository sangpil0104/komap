"""Write Korean Markdown report tables and verify a PNG/Markdown-only ZIP."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT=Path('/Data3/sppark/komap')
HERE=Path(__file__).resolve().parent
ANALYSIS=HERE/'outputs/analysis_20261007_1'
RUN=ROOT/'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'
PACKAGE=ROOT/'results/model349_report_materials_20261007_1'
TABLES=PACKAGE/'tables'
EDA=json.loads((ANALYSIS/'eda.json').read_text())
MET=json.loads((ROOT/'results/model_results_0930_1/metrics.json').read_text())
ABL=json.loads((ANALYSIS/'ablations.json').read_text())
CAT=sorted(json.loads((ANALYSIS/'figure_catalog.json').read_text()),key=lambda r:r['file'])
NAMES=['Al3Ni','Primary Si','Al','Eutectic Si']
SPLITS=['train','valid','test']
TABLE_CATALOG=[]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x).replace('\n','<br>') for x in row)+' |' for row in rows])


def write(name,title,parts,section):
    body='# '+title+'\n\n'+'\n\n'.join(parts)+'\n'
    with (TABLES/name).open('x',encoding='utf-8') as f:f.write(body)
    TABLE_CATALOG.append({'file':name,'title':title,'section':section})


def figure_link(n):
    r=next(r for r in CAT if r['file'].startswith(f'{n:02d}_'))
    return f"[{r['title']}](../figures/{r['file']})"


def data_tables():
    rows=[]
    for s in SPLITS:
        d=EDA['summary'][s]
        rows.append([s,d['images'],len(d['specimens']),f"{d['total_pixels']:,}",'있음' if s!='test' else '없음',f"{d['gray_mean_mean']:.3f}",f"{d['gray_std_mean']:.3f}"])
    sizes=sorted({(r['width'],r['height']) for r in EDA['records']},key=lambda x:x[0]*x[1])
    dimensions=[[f'{w} × {h}']+[sum(r['split']==s and r['width']==w and r['height']==h for r in EDA['records']) for s in SPLITS] for w,h in sizes]
    overlaps=[[k,len(v),', '.join(v)] for k,v in EDA['overlap'].items()]
    write('01_dataset_summary.md','데이터 구성 및 분할',[
        table(['분할','이미지 수','고유 시편 ID 수','전체 픽셀 수','정답 마스크','평균 밝기의 이미지 평균','밝기 표준편차의 이미지 평균'],rows),
        '밝기는 PIL grayscale의 0–255 값이다. 영상별 밝기 통계는 이미지를 동일 가중치로 평균했다. 시편 ID는 파일명 앞 A번호로 추정한 식별자이다.',
        '**해상도별 이미지 수**',table(['너비 × 높이 (pixel)','train','valid','test'],dimensions),
        '**분할 간 시편 ID 교집합**',table(['분할 쌍','공유 ID 수','공유 ID'],overlaps),
        '제공된 split을 그대로 사용했다. B4 학습 내부의 그룹 4-fold와 최종 train/valid의 시편 분리는 별개이다. 파일명 기준 train/valid 공유 ID가 있으므로 완전히 독립된 시편 평가라고 기재하지 않는다. 원본 해상도로 물리적 배율이나 길이 단위를 추정하지 않는다.',
        '그림: '+', '.join(figure_link(n) for n in [1,2,5,6]),
    ],'1-3 EDA')
    rows=[]
    for c,n in enumerate(NAMES):
        for s in ['train','valid']:
            d=EDA['summary'][s]
            rows.append([n,s,f"{d['pixel_fraction'][c]*100:.4f}",f"{d['image_fraction_mean'][c]*100:.4f}",
                         f"{d['image_fraction_median'][c]*100:.4f}",f"{d['image_fraction_q25'][c]*100:.4f}–{d['image_fraction_q75'][c]*100:.4f}",
                         f"{d['present_images'][c]}/{d['images']}"])
    absent=[[s,c,', '.join(v)] for s in ['train','valid'] for c,v in EDA['summary'][s]['absent_images'].items() if v]
    write('02_class_distribution.md','클래스 불균형과 이미지별 면적 분포',[
        table(['클래스','분할','전체 픽셀 비율 (%)','이미지별 면적 비율 평균 (%)','중앙값 (%)','Q1–Q3 (%)','정답에 존재하는 이미지'],rows),
        '전체 픽셀 비율 = 해당 클래스 총 픽셀 수 / split 전체 픽셀 수. 이미지별 면적 비율 평균 = 각 이미지의 클래스 면적 비율을 동일 가중치로 평균. 큰 이미지의 영향을 받는 정도가 다르므로 둘을 구분한다. 두 통계 모두 데이터 구성비이며 IoU 성능 지표가 아니다.',
        '**정답에서 클래스가 없는 이미지**',table(['분할','클래스','이미지'],absent),
        'Al이 가장 많은 픽셀을 차지하고 Eutectic Si가 가장 적다. 클래스별 평가와 모델 조합을 검토할 배경 자료이며, 불균형 자체가 특정 설계의 성능 향상을 입증하지는 않는다.',
        '그림: '+figure_link(3)+', '+figure_link(4),
    ],'1-3 EDA')


def method_tables():
    write('03_preprocessing_and_augmentation.md','입력 표현·전처리·증강',[
        table(['모델','입력 표현','크기 설정','학습 통계 범위','평가 시 처리'],[
            ['Model322','grayscale 동일 채널 3회 복제','3배 확대','(g−0.449)/0.226','전체 영상 forward + D4'],
            ['Model326','Raw / Z-score / CDF','3배 확대','확대한 전체 영상의 μ·σ·CDF','전체 영상 forward + D4'],
            ['Model329','Raw / Z-score / CDF','너비 4000','크기를 맞춘 전체 영상의 μ·σ·CDF','전체 영상 forward + D4'],
            ['Model269','Raw / Z-score / rank-CDF','scale 1; 최소 변이 패치보다 작으면 확대','전체 영상 표현','1024 타일, stride 576, D4'],
            ['Model336 / 347','RGB + ImageNet 정규화','학습 입력 scale 2','고정 채널별 ImageNet 통계','native 256 타일을 512로 확대; flip4'],
        ]),
        '**B4의 Raw/Z-score/CDF 정의**',
        r'\(g\in[0,1]\), \(X_{raw}=(g-0.5)/0.25\), \(X_z=\operatorname{clip}((g-\mu)/(\sigma+10^{-6}),-4,4)/3\), \(X_{CDF}=(CDF(g)-0.5)/0.288675\).',
        'Model326·329는 전체 크기 조정 영상의 통계를 미리 구하고, 증강·패치 추출 후 입력 픽셀을 해당 통계로 변환한다. CDF는 256단계 밝기 histogram 누적분포이다. Model269 계열은 rank 기반 CDF 구현을 사용하므로 B4 함수를 그대로 대체하지 않는다.',
        '**B4 학습 증강**',table(['연산','설정'],[
            ['Affine','scale 0.7–1.4, 회전 −180°–180°, p=0.7'],['Padding / crop','reflect padding 후 1536×1536 random crop'],
            ['D4','직각 회전 및 반사'],['Brightness / contrast','limit 0.3 / 0.4, p=0.8'],
            ['Gamma','60–160, p=0.5'],['Blur 또는 noise','Gaussian blur 3–5 또는 noise std 0.01–0.05, OneOf p=0.3'],
        ]),
        '**HRNet 학습 증강**',table(['연산','설정'],[
            ['수평·수직 반전 / 90° 회전','각 p=0.5'],['Affine','scale 0.8–1.2, p=0.3'],
            ['Brightness / contrast','각 limit 0.15, p=0.5'],['Gamma','85–115, p=0.3'],
            ['Elastic 또는 grid distortion','OneOf p=0.3'],['Noise / blur','noise std 0.02–0.08, p=0.2 / blur 3–5, p=0.15'],
        ]),
        '기하 변환은 입력과 정답의 위치가 일치하도록 함께 적용한다. 크기 조정 시 클래스 마스크에는 nearest-neighbor 보간을 사용한다. Validation/test에서는 학습용 확률 증강을 사용하지 않고 지정된 TTA를 적용한다.',
        '그림: '+figure_link(7)+', '+figure_link(8),
    ],'2-1 전처리 방법 / 2-2 절차')
    write('04_model_components.md','모델 구성과 설계 의도',[
        table(['구성','Encoder / decoder','특징','최종 결합 역할'],[
            ['기본 확률','ResNet34 / compact U-Net++ 및 ASPP 계열','97·98·109, pairwise, 고해상도·스타일·TENT 분기','기본 확률'],
            ['Model269','ResNet34 / compact U-Net++','TeamResidualASPP + stride-1 native stem','Eutectic Si'],
            ['Model322','EfficientNet-B4 / SMP U-Net','gray3, 4-fold','Al'],
            ['Model326','EfficientNet-B4 / SMP U-Net','Raw/Z/CDF, 4-fold','초기 Primary Si·Al3Ni'],
            ['Model329','EfficientNet-B4 / SMP U-Net','너비 4000, 이미지 균등 샘플링, 4-fold','Primary Si·Al3Ni'],
            ['Model336','HRNet-W48 / SMP U-Net++','학습용 1채널 boundary head','Primary Si·Al·Eutectic Si'],
            ['Model347','Model336과 동일','클래스 가중 CE로 fine-tuning','마지막 Al 확률 보완'],
        ]),
        '모든 전문가가 네 클래스를 출력하며 위 역할은 선택된 결합 채널을 뜻한다. B4 계열은 총 12개 fold checkpoint이다. 기본 확률 자체도 복수 모델·추론 분기를 포함하므로 전체 시스템을 단일 네트워크로 기술하지 않는다.',
        table(['설계 요소','구조 또는 의도','현재 근거의 범위'],[
            ['RGB residual ASPP','1×1, dilation 6/12/18의 3×3, global pooling; 각128채널 → 640 → 512','소스에서 구조 확인'],
            ['표현 변환 ASPP','1×1, dilation 2/4/8의 3×3, global pooling','소스에서 구조 확인'],
            ['Residual scale','x + s·projection, s=0 초기화','추가 시 기존 경로 출력을 유지하도록 설계'],
            ['Native stem','입력 크기 Conv 3→32→32 + zero-init Conv 32→32를 head 내부에 더함','독립 재학습 ablation 미수행'],
            ['Boundary head','decoder 특징에서 3×3 Conv → 1채널 경계; 학습에만 사용','독립 재학습 ablation 미수행'],
            ['표현 다양성','절대 밝기, 상대 밝기, 밝기 순위 정보를 함께 제공','입력 예시 확인; 인과적 기여 분리 실험 미수행'],
        ]),
        '그림: '+', '.join(figure_link(n) for n in [9,10,11]),
    ],'3-1 배경/알고리즘 / 3-2 모델링')

    args={k:json.loads((RUN/'outputs'/d/'args.json').read_text()) for k,d in {
        '322':'model322_team_b4_group_oof/fold1','326':'model326_hybrid_fullstats_group_oof/fold1',
        '329':'model329_fullstats_width4000_uniform_group_oof/fold1','336':'model336_hrnet_best',
        '347':'model347_hrnet_inverse_ce_finetune'}.items()}
    rows=[]
    for k,a in args.items():
        b4=k in ['322','326','329']
        rows.append([f'Model{k}',a['crop'] if b4 else a['patch_size'],a['bs'] if b4 else a['batch_size'],
                     a['accum'] if b4 else a['accumulation_steps'],f"{a['iters']:,} steps/fold" if b4 else f"{a['epochs']} epochs",
                     a['lr'],a['wd'] if b4 else a['weight_decay'],a['seed'],'last.pt'])
    rows.append(['Model269',1024,2,1,'4 epochs (Model257 이후)', 'encoder 2.5e-6 / decoder 7.5e-6 / native 5e-5','1e-4',269,'best.pt'])
    write('05_training_and_inference.md','학습·추론 설정',[
        table(['모델','패치 크기','Batch','누적','학습 길이','Learning rate','Weight decay','Seed','사용 checkpoint'],rows),
        '모두 AdamW를 사용한다. B4는 300-step warm-up을 곱한 cosine schedule, HRNet은 epoch 단위 cosine schedule을 사용한다. B4의 모든 fold 인자에서 seed=0이지만 일부 worker seed는 시간에 의존하여 재학습의 완전한 결정성이 보장되지는 않는다.',
        table(['계열','학습 손실','초기화 / 추가 설정'],[
            ['B4 322/326/329','CE + Dice','ImageNet encoder; Lovasz/clDice/EMA 미사용'],
            ['HRNet336','0.5 CE + 0.5 Dice + boundary loss','ImageNet encoder; CE 클래스 가중치 없음'],
            ['HRNet347','0.5 weighted CE + 0.5 Dice + boundary loss','336 last.pt에서 시작; encoder LR 배율0.3'],
            ['Model269','weighted CE + Dice','Model257 best.pt에서 연속 fine-tuning'],
        ]),
        'Boundary loss = positive-reweighted BCE + Dice. Model347 CE 가중치는 팀 순서 [Al, Al3Ni, Primary Si, Eutectic Si]에서 [0.22906544, 0.49862612, 1.39166570, 1.88064275]이다.',
        table(['분기','추론 크기/타일','TTA','확률 결합'],[
            ['기본 97/98/109','640/512/512, stride512/384/384','원본·수평·수직·90°회전의 4개','uniform 타일 결합; 저장 캐시 fp16'],
            ['고해상도·pairwise','고해상도768/1024; pairwise768','D4 8개','Gaussian 타일 가중; 고해상도768:1024=0.8:0.2'],
            ['B4 322/326/329','전체 영상; 3배 또는 너비4000','D4 8개','변환 복원 후 4-fold 평균'],
            ['HRNet336/347','native256 → 입력512; overlap170, stride86','수평·수직 반전 조합4개','Gaussian window로 타일 결합'],
            ['Model269','1024 타일, stride576, scale1','D4 8개','Gaussian 타일 결합 및 원본 크기 복원'],
        ]),
        '기본 고해상도 분기의 TENT는 입력 영상마다 BN affine 파라미터를 entropy 최소화로 5 step 적응시키는 원래 추론 절차이다(lr=1e-5). 실제 test 출력은 기존 체크포인트와 고정 조합을 사용했으며 test 정답을 사용하지 않았다.',
    ],'3-2 모델링 방법')

    write('06_fusion_and_postprocessing.md','클래스별 결합과 후처리',[
        '클래스 순서는 [Al3Ni, Primary Si, Al, Eutectic Si]. 팀 순서 [Al, Al3Ni, Primary Si, Eutectic Si]의 확률을 [1,2,0,3] 채널 인덱스로 재배열한다.',
        table(['단계','Al3Ni','Primary Si','Al','Eutectic Si'],[
            ['Model339','326 / 0.70','326 / 1.00','322 / 0.60','269 / 0.70'],
            ['Model341','329 / 0.50','329 / 1.00','—','—'],['Model346','—','336 / 0.25','336 / 0.75','336 / 0.75'],
            ['Model349','—','—','347 / 0.25','—'],
        ]),
        r'해당 클래스만 \(\widetilde P_c=(1-\gamma)P_c+\gamma Q_c\)로 교체한 뒤, 네 클래스 전체를 \(P_k=\widetilde P_k/\sum_j\widetilde P_j\)로 정규화한다. 다른 클래스도 재정규화의 영향을 받으므로 각 단계 계수를 곱해 전체 가중치로 해석하지 않는다.',
        'Model339 내부 순서는 Primary Si → Al3Ni → Eutectic Si → Al. 이후 Model329, Model336, Model347 순서로 각 클래스 인덱스의 0이 아닌 계수를 처리한다.',
        table(['처리','정확한 설정'],[
            ['밝기 prior','Z-score prior0.85 + CDF prior0.15; 적응형 차이 항은 Al3Ni에 적용'],
            ['Logit 보정','log(max(P,1e-7)) + 0.09×prior; prior_scale1.25, beta_scale0.80'],
            ['클래스 bias','영상 통계에 따른 기본 bias + al_delta0.00, eut_delta+0.15'],
            ['Primary Si 공간 필터','Gaussian(sigma0.50)0.50 + bilateral0.50'],
            ['Eutectic Si 공간 필터','원래 logit0.75 + bilateral0.25'],
            ['Bilateral','Z-score guide, radius2, spatial sigma1.0, range sigma0.60'],
            ['연결 성분 정리','Primary Si의 128픽셀 이하 성분을 가장 점수 높은 다른 클래스로 재할당'],
            ['영상 면적 규칙','정리 후 Primary Si 비율이 1% 이하이면 남은 해당 픽셀도 다른 클래스로 재할당'],
        ]),
        '작은 Primary Si가 제거될 수 있는 규칙이므로 추론 방법의 일부로 공개한다. 영상별 source 선택·bias 조건은 source_snapshot의 highres1024_search.py 및 hard_source_search.py에 보존되어 있다.',
        '그림: '+figure_link(9)+', '+figure_link(12),
    ],'3-2 모델링 방법')


def result_tables():
    rows=[];last=None
    for mid,folder,desc in [('339','model339_joint_gamma_route_search','초기 전문가 결합'),('341','model341_model329_joint_route_search','Model329 추가'),('346','model346_model336_into_341_route','Model336 추가'),('349','model349_model347_into_346_route','Model347의 Al 확률 추가')]:
        d=json.loads((RUN/'outputs'/folder/'results.json').read_text())
        score=d['best_image_empty_zero']['image_mean_miou'] if 'best_image_empty_zero' in d else d['raw_best_valid_image_mean_miou_empty0']
        rows.append([f'Model{mid}',desc,f'{score*100:.6f}%', '—' if last is None else f'{(score-last)*100:+.6f}%p']);last=score
    cv=MET['route']['grouped_cv_vs_model346']
    write('07_stage_performance.md','조합 단계별 성능',[
        table(['단계','변경','Validation image mIoU','직전 표 단계 대비'],rows),
        '모두 동일 로컬 실행의 validation20, image-wise 4-class mIoU(empty=0)이다. 순차 개발 기록이며 각 요소를 독립적으로 제거해 재학습한 실험은 아니다.',
        table(['보조 분석','결과'],[
            ['Model347 계수 선택의 grouped CV','19개 시편 그룹 중 한 그룹씩 제외하여 계수 선택'],
            ['Model346 대비 평균 변화',f"{cv['image_mean_delta_pp']:+.6f}%p"],
            ['그룹 bootstrap 95% 구간',f"[{cv['group_bootstrap_95ci_delta_pp'][0]:+.6f}, {cv['group_bootstrap_95ci_delta_pp'][1]:+.6f}]%p"],
        ]),
        '기본 모델·선행 조합은 이미 validation으로 개발·선택되었으므로 이 보조 검증은 전체 파이프라인의 독립 평가가 아니다. Model347 추가의 +0.017012%p를 안정적으로 입증된 향상이라고 단정하지 않는다.',
        '그림: '+figure_link(13),
    ],'3-3 성능')
    means=np.array([[r['class_iou'][c] for c in NAMES] for r in MET['images']]).mean(0)
    rows=[[r['image'],f"{r['image_miou']*100:.6f}"]+[f"{r['class_iou'][c]*100:.6f}" for c in NAMES] for r in MET['images']]
    rows.append(['20장 평균',f"{MET['image_miou']*100:.6f}"]+[f'{x*100:.6f}' for x in means])
    write('08_validation_scores.md','Validation 이미지별·클래스별 점수',[
        table(['이미지','image mIoU (%)']+[f'{c} IoU (%)' for c in NAMES],rows),
        '각 이미지의 네 클래스 IoU 평균을 먼저 계산하고, 20장에 대해 동일 가중치로 평균한다. 정답·예측 모두에 없는 클래스는 IoU=0이다. 100%−image mIoU는 픽셀 오류율이 아니다.',
        '현재 EDA에서 A71_step2_sample1의 정답에 Primary Si가 없는 것을 확인했다. 해당 지표의 0을 모두 검출 실패로 해석하지 않는다.',
        '그림: '+', '.join(figure_link(n) for n in [14,15,16,18]),
    ],'3-3 성능')
    rows=[[r['label'],f"{r['image_miou']*100:.6f}",f"{r['delta_pp']:+.6f}"]+[f'{v*100:.6f}' for v in r['class_iou']] for r in ABL['cases']]
    write('09_frozen_probability_comparisons.md','저장 확률 기반 후처리·결합 비교',[
        '2026-10-07에 기존 validation 확률 캐시만 사용해 추가 산출한 사후 분석이다. 새로운 학습, checkpoint 선택, 계수 탐색을 수행하지 않았다. 원래 Model349 조건이 기존 마스크20장과 모두 픽셀 단위로 일치함을 확인했다.',
        table(['조건','image mIoU (%)','원래 대비 (%p)']+NAMES,rows),
        '클래스 열의 단위는 %이며 이미지별 클래스 IoU의 평균이다.',
        table(['조건','변경 정의'],[[r['label'],r['definition']] for r in ABL['cases']]),
        '균등 평균에서 기본 확률은 이미 내부 앙상블이다. 따라서 이 조건은 모든 신경망 checkpoint에 동일 가중치를 준 결과가 아니라, 기본 확률 및 6개 전문가 확률원의 평균이다. B4 각 확률원에는 4-fold 평균이 들어 있다. 평균 조건에도 원래 후처리와 계수를 그대로 사용하여 최적화하지 않았다.',
        '**해석**',
        '모든 최종 보정을 제거하면 원래 Model349보다 1.220346%p 낮다. 밝기 prior를 제거하면 0.424717%p, 7개 확률원을 균등 평균하면 0.454289%p 낮다. 각 차이는 이 고정 설정과 검증20장의 관측 결과이다.',
        '공간 필터만 제거한 점수는 80.068934%로 원래보다 +0.000135%p 높다. 이 자료에서는 공간 필터의 개선 효과를 주장하기 어렵다. 이 매우 작은 차이로 새로운 우수 모델을 확정하지 않으며, 기존 보고 점수80.068799%와 구분한다.',
        '**재학습이 있어야 분리 가능한 비교**',table(['비교','현재 상태','필요 조건'],[
            ['Raw만 입력 vs Raw/Z/CDF','인과적 비교 미수행','같은 split·backbone·seed·예산으로 입력만 바꿔 학습'],
            ['ASPP 유무','인과적 비교 미수행','같은 학습 조건에서 모듈만 변경'],
            ['native stem 유무','인과적 비교 미수행','동일 시작 checkpoint와 학습 예산'],
            ['boundary loss 유무','인과적 비교 미수행','동일 HRNet 학습 조건에서 보조 손실만 변경'],
        ]),
        '위 미수행 항목의 점수는 만들어 넣지 않았다. 새 학습은 사용자가 시작한다.',
        '그림: '+figure_link(17),
    ],'3-3 성능 / 추가 분석')


def reproducibility_and_captions():
    freeze={}
    for line in (RUN/'environment.freeze.txt').read_text().splitlines():
        if '==' in line:
            k,v=line.split('==',1);freeze[k]=v
    libs=['torch','torchvision','segmentation_models_pytorch','timm','numpy','scipy','opencv-python-headless','albumentations','pillow','tqdm']
    write('10_evaluation_and_reproducibility.md','평가 조건·재현 정보·제출 산출물',[
        table(['항목','내용'],[
            ['Model349 실행 ID',RUN.name],['보고된 점수','Validation20 image mIoU 80.068799%'],
            ['평가 방식','이미지별4클래스 평균 → 20장 평균; union=0이면IoU=0'],
            ['데이터','train70, valid20, test10; 제공된 split 사용'],
            ['모델 선택','일부 checkpoint와 최종 조합 선택에 valid 사용'],
            ['B4 내부 검증','시편 그룹 기반4-fold; 최종 valid 확률은4-fold 평균'],
            ['재현 환경','Conda komap; 실제 학습 당시 freeze는 아래 표'],
            ['검증 마스크','기존 결과20장; 새로 만든 표의 원래 조건과 픽셀 일치 확인'],
            ['Test 예측','기존 모델·고정 조합으로10장 추론; test 정답 사용 없음'],
            ['공개된 test 성능','정답이 없어 계산하지 않음'],
            ['전체 파라미터 수 / FLOPs','별도 계측 자료 없음'],
            ['독립 test 또는 외부SOTA','이 자료 묶음에서 검증하지 않음'],
        ]),
        table(['라이브러리','기록된 버전'],[[k,freeze[k]] for k in libs]),
        '보고서에서는 “본 비교 조건의 validation 성능”으로 범위를 명시한다. 외부 SOTA를 주장하려면 동일 데이터·분할·지표·모델 선택 조건의 비교 근거가 필요하다. Validation을 미사용 독립 test로 표현하지 않는다.',
        '원본 배포 설명의79.9095189%는 과거 기록이다. 이 실행의80.068799%와 혼용하지 않는다. 새로 산출한 저장확률 비교표도 원래 모델의 보고 점수와 구분한다.',
        '**자료 출처**',table(['근거','프로젝트 내 위치'],[
            ['완료 실행',f'`{RUN.relative_to(ROOT)}`'],
            ['실행 소스',f'`{(RUN/"source_snapshot").relative_to(ROOT)}`'],
            ['학습 인자','위 실행 outputs/각 모델/args.json 및 launcher_snapshot/pipeline_tasks.json'],
            ['공식 로컬 결과','`results/model_results_0930_1/metrics.json`'],
            ['기존 상세 설명','`results/model349_detailed_report_20261007.md`'],
            ['기존 valid/test ZIP','`results/model349_valid_test_predicted_masks_20261007_1.zip`'],
            ['이 자료 생성 코드','`experiments/model349_report_materials_v1/`'],
        ]),
        '이 ZIP에는 그래프·구조도와 Markdown 표만 담았다. 기존 valid/test 마스크와 비교 이미지는 사용자가 보유한 파일을 사용하면 된다. 개인정보·팀명·접수번호·수상경력은 제공되지 않아 채우지 않았다.',
    ],'3-2 방법 / 3-3 성능 / 재현 정보')
    rows=[[f"{r['file'][:2]}",r['section'],f"[{r['title']}](../figures/{r['file']})",r['caption']] for r in CAT]
    write('11_figure_captions_and_page_plan.md','그림 캡션 및 10페이지 배치안',[
        '제출 양식의 필수 소제목을 유지하고 아래 자료를 골라 배치한다. 모든 보조 그림을 본문에 넣기보다 결과를 설명하는 대표 자료를 선택한다.',
        table(['페이지','내용','우선 자료'],[
            ['1','팀 정보·배경·아이디어·파급효과','배경과 기대효과는 각5줄 이내; 팀 정보는 직접 기입'],
            ['2–3','1-1 핵심 내용 / 1-2 배경 / 1-3 EDA','그림01·03·04·05, 표01·02'],
            ['4','2-1 전처리 방법 / 2-2 절차','그림07·08, 표03'],
            ['5–6','3-1 알고리즘 / 3-2 모델링 방법','그림09·12, 표04·05·06; 상세도10·11은 공간에 따라 선택'],
            ['7','3-3 성능: 점수와 비교','그림13·14, 표07·08'],
            ['8','3-3 성능: 개별 결과','기존 보유 comparison과 그림15 또는16'],
            ['9','3-3 성능: 추가 분석','그림17, 표09'],
            ['10','한계·재현·test 산출물 안내','표10, 필요하면 그림06·18'],
        ]),
        '**그림별 권장 캡션**',table(['번호','삽입 위치','그림','캡션'],rows),
        '이미지는300dpi PNG이다. 입력 표현 그림은 실제 입력 함수로 계산했고 구조도는 소스에 근거해 그렸다. 통계 그래프의 숫자는 동봉된 표와 대조할 수 있다. 기존 예측 마스크·comparison은 이 ZIP에 중복 수록하지 않았다.',
    ],'작성 가이드')


def package():
    # Portable internal links and explicit provenance, without embedding old masks.
    readme=['# Model349 보고서 추가 자료',
            '**PNG 그림18개와 Markdown 표·해설11개**를 모은 자료이다. Model349 원래 점수는 validation20 image mIoU **80.068799%**이다.',
            '데이터 통계는 제공된 train70/valid20/test10에서 계산했고, test 클래스 마스크는 사용하지 않았다. 추가 비교는 저장된 validation 확률로만 계산했다. 기존 valid/test 마스크·comparison 및 기존 상세 보고서는 중복으로 담지 않았다.',
            '**그림 목록**',table(['파일','내용','권장 위치'],[[f"[{r['file']}](figures/{r['file']})",r['title'],r['section']] for r in CAT]),
            '**표 목록**',table(['파일','내용','권장 위치'],[[f"[{r['file']}](tables/{r['file']})",r['title'],r['section']] for r in TABLE_CATALOG]),
            '그림별 캡션과10페이지 배치안은 [표11](tables/11_figure_captions_and_page_plan.md)에 있다. 수치표의 단위(%, %p)와 전체 픽셀 구성비/이미지별 평균의 차이를 유지하여 인용한다.',
            'PNG는300dpi이며 한글 글꼴은NanumGothic을 사용했다. 그림18의60%는 지표 설명용 예시이다. 새로 학습한 모델의 결과를 포함하지 않는다.',
            '저장확률 비교에서 공간 필터 제거가 +0.000135%p 높게 나타난 사실도 그대로 수록했다. 원래 모델 점수나 checkpoint를 바꾸지 않았다.',
            '분석·생성 코드: `experiments/model349_report_materials_v1/analyze.py`, `figures.py`, `tables_and_package.py`. 모든 스크립트는 Conda `komap`에서 실행했다.',
           ]
    (PACKAGE/'README.md').write_text('\n\n'.join(readme)+'\n')
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    checks=[]
    for p in tqdm(files,desc='Validate report package',unit='file'):
        assert p.suffix in ['.png','.md'],p
        if p.suffix=='.png':
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                assert min(im.size)>=1200,(p,im.size)
                assert im.info.get('dpi',(0,0))[0]>299
        else:
            text=p.read_text()
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if not link.startswith(('http','/','#')):assert (p.parent/link).exists(),(p,link)
            assert '@CLASS_TABLE@' not in text
        checks.append([p.relative_to(PACKAGE).as_posix(),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
    assert len(list((PACKAGE/'figures').glob('*.png')))==18
    assert len(list(TABLES.glob('*.md')))==11
    provenance=['# 파일 검증 및 출처',
                '그림18개와표11개, README의SHA-256을 기록했다. 이 파일 자체는 목록에서 제외한다.',
                table(['파일','크기(bytes)','SHA-256'],checks),
                '분석 기준 실행: `'+str(RUN)+'`.',
                '모든 원래 확률결합·후처리 조건에서 기존 valid 마스크20장과 픽셀 일치를 검증했다. 입력 마스크에서알려지지 않은 색상이 없는지 확인했다. Global mIoU는 계산하지 않았다.',
                '확률 캐시 입력 파일의 해시 기록은 프로젝트 내 `experiments/model349_report_materials_v1/outputs/analysis_20261007_1/sources.json`에 보존했다.',
               ]
    (PACKAGE/'PROVENANCE.md').write_text('\n\n'.join(provenance)+'\n')
    zip_path=PACKAGE.with_suffix('.zip')
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in tqdm(files,desc='Archive PNG and Markdown',unit='file'):
            z.write(p,p.relative_to(PACKAGE).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert len(z.namelist())==31
        for p in files:assert hashlib.sha256(z.read(p.relative_to(PACKAGE).as_posix())).digest()==hashlib.sha256(p.read_bytes()).digest()
    result={'zip':str(zip_path),'bytes':zip_path.stat().st_size,'png':18,'table_md':11,'guide_md':2,
            'sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'verified':True}
    (ANALYSIS/'package_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    TABLES.mkdir(exist_ok=False)
    for fn in tqdm([data_tables,method_tables,result_tables,reproducibility_and_captions],desc='Create Markdown tables',unit='group'):fn()
    package()
