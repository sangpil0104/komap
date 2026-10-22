"""Manuscript grounded in the existing template, audited local results and pinned GitHub."""
from pathlib import Path
import json,re,sys
from analyze import PACKAGE,ROOT,RESULT,EXP,RUN,CLASSES,PALETTE,read,write
from render import render

M=read(PACKAGE/'evidence/audited_metrics.json');EDA=read(PACKAGE/'evidence/eda.json')
C=read(PACKAGE/'evidence/pixel_changes.json');T=read(PACKAGE/'evidence/runtime.json')
F={r['id']:r for r in read(PACKAGE/'figures/catalog.json')}
GITHUB='https://github.com/CNUKoMaP/lhs/blob/ac6252bef323bdba7482f89a61420312310901e9/'
refs={
 'S01':dict(title='프로젝트 지침 및 원본 결과서 양식',url='evidence/template_text.txt',note='원본 .doc의 필수 목차와 평가 정의. 평가 정의는 프로젝트 AGENTS.md와 일치한다.'),
 'S02':dict(title='실제 데이터 EDA 및 입력 해시',url='evidence/eda.json',note='train70/valid20/test10 입력 파일에서 새로 계산. test 정답을 사용하지 않았다.'),
 'S03':dict(title='로컬 Fusion805 재학습 결과',url='evidence/local_result_report.md',note='results/model_results_20261009_2; 결과·계획·출처 사본을 evidence/에 보존.'),
 'S04':dict(title='R(exh14)의 실제 실행 코드',url='evidence/execution_source/frozen/infer_lite.py',note='B4/HRNet TTA off, native none, highres 768 D4/TENT5 등 실행 인자의 기준.'),
 'S05':dict(title='E 전문가 추론 및 family 결합',url='evidence/execution_source/expert_worker.py',note='실제 학습 소스는 fusion_source/model397_reproduction_bundle/. 결합은 features.py.'),
 'S06':dict(title='F의 OOF 입력·학습 소스',url='evidence/execution_source/fusion_source/refiner/train_residual.py',note='features.py, residual_model.py, gray_input.py와 refiner_training_receipt.json을 함께 확인.'),
 'S07':dict(title='저장 마스크의 독립 재채점',url='evidence/audited_metrics.json',note='R/Stage1/Stage2, 20장 × 4클래스 × 3단계 검증; E는 저장 확률 argmax. pixel_changes.json 포함.'),
 'S08':dict(title='GitHub fusion805 재현 및 대표 후보 설명',url=GITHUB+'docs/model349_fusion805_reproduction.md',note='ac6252b 고정. 동봉 사본: evidence/github_model349_fusion805_reproduction.md.'),
 'S09':dict(title='GitHub stage1 fresh 추론 결과',url=GITHUB+'docs/model349_stage1_fresh_report.md',note='고정 가중치 추론 재현 결과이며 로컬 재학습 점수와 구분. 동봉 사본 포함.'),
 'S10':dict(title='실제 작업 시간 및 학습 기록',url='evidence/runtime.json',note='원본 events.jsonl, status.json, finished.json, F receipt를 기준으로 집계.'),
 'S11':dict(title='현재 프로젝트 결과 문서 목록',url='tables/project_result_inventory.csv',note='실제 존재하는 model_results_*/동명.md를 집계. 파일명 끝 숫자를 모델 번호로 해석하지 않는다.'),
 'S12':dict(title='실행 계획 및 학습 인자',url='evidence/local_plan.json',note='학습 27개, 전체 작업 43개. 체크포인트 manifest 및 환경 freeze 별도 동봉.'),
 'S13':dict(title='보정기의 OOF 학습 입력 manifest',url='evidence/refiner_oof_manifest.json',note='70장 각각의 hold-out fold·입력 확률·정답 해시. valid_read=false, test_read=false.'),
 'P01':dict(title='U-Net: Convolutional Networks for Biomedical Image Segmentation',url='https://arxiv.org/abs/1505.04597',note='Ronneberger et al., 2015. Encoder–decoder와 skip 연결의 배경.'),
 'P02':dict(title='UNet++: A Nested U-Net Architecture for Medical Image Segmentation',url='https://arxiv.org/abs/1807.10165',note='Zhou et al., 2018. 중첩 skip 연결의 배경.'),
 'P03':dict(title='Deep Residual Learning for Image Recognition',url='https://arxiv.org/abs/1512.03385',note='He et al., 2015. ResNet의 배경.'),
 'P04':dict(title='EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks',url='https://arxiv.org/abs/1905.11946',note='Tan and Le, 2019. EfficientNet의 배경.'),
 'P05':dict(title='Deep High-Resolution Representation Learning for Visual Recognition',url='https://arxiv.org/abs/1908.07919',note='Wang et al., 2019. HRNet의 배경.'),
 'P06':dict(title='DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs',url='https://arxiv.org/abs/1606.00915',note='Chen et al., 2016. Atrous convolution과 다중 스케일 문맥의 배경. 본 프로젝트가 논문의 CRF를 사용한다는 뜻은 아니다.'),
 'P07':dict(title='Tent: Fully Test-time Adaptation by Entropy Minimization',url='https://arxiv.org/abs/2006.10726',note='Wang et al., 2020. 정답 없이 entropy를 최소화하는 test-time adaptation의 배경.')}
B=[]
def p(text):B.append(dict(type='p',text=text))
def h(text,level=2,newpage=False):B.append(dict(type='heading',text=text,level=level,newpage=newpage))
def eq(text,markdown):B.append(dict(type='equation',text=text,markdown=markdown))
def tab(id,title,headers,rows,note=''):B.append(dict(type='table',id=id,title=title,headers=headers,rows=rows,note=note))
def fig(id):B.append(dict(type='figure',**F[id]))
def pct(x,n=6):return f'{x*100:.{n}f}%'
def score(k):return M[k]['image_miou']
def delta(a,b):return (score(b)-score(a))*100
def img(k,n):return next(r for r in M[k]['images'] if r['image']==n)

B.extend([dict(type='title',text='모델링 결과서'),dict(type='subtitle',text='고탄성 알루미늄 합금 이미지 상 분할 인공지능 모델'),dict(type='subtitle',text='Model349 · Fusion805 | 분석·모델링 통합 확장본')])
p('작성 기준: 2026년 10월 9일, 한국 시간. 로컬 결과 model_results_20261009_2와 GitHub commit ac6252bef323bdba7482f89a61420312310901e9를 기준으로 작성했다.')
identity=read(PACKAGE/'evidence/identity.json') if (PACKAGE/'evidence/identity.json').exists() else {}
tab('cover','기본 정보',['항목','내용'],[
 ['접수번호',identity.get('registration','미제공(제출 전 기입)')],['팀명',identity.get('team','미제공(제출 전 기입)')],
 ['최근 3년 공모전 수상경력',identity.get('awards','미제공(제출 전 기입)')],
 ['주요 활용 라이브러리','PyTorch, torchvision, segmentation-models-pytorch, timm, Albumentations, NumPy, Pillow, OpenCV, tqdm'],
 ['대표 결과',f'R+E: image mIoU {pct(score("stage1"))} / valid20'],
 ['추가 비교 결과',f'R+E+F: image mIoU {pct(score("stage2"))} / valid20']],'표지 인적 사항은 저장소 자료만으로 확인할 수 없어 임의로 채우지 않았다. [S01, S03]')
h('배경 및 아이디어 소개(요약)',level=1)
h('아이디어 제안 배경 — 5문장 요약')
p('알루미늄 합금 현미경 영상을 Al₃Ni, Primary Si, Al, Eutectic Si의 네 상으로 분할하는 것을 목표로 한다. 제공 학습 영상이 70장이므로 사전학습 특징, 패치 학습, 증강과 시편 단위 분할을 조합했다. 영상별 밝기·해상도 차이와 작은 상의 경계를 함께 다루기 위해 서로 다른 입력 표현과 신경망 계열을 활용했다. 기존 Model349의 고정 추론 패키지 R에 독립 전문가 E의 클래스 선택 규칙을 결합하고, OOF 예측으로 학습한 작은 잔차 보정기 F도 평가했다. 현재 재학습에서는 R+E가 80.236507%, R+E+F가 80.267268%의 valid20 image mIoU를 기록했다. [S02–S08]')
h('파급효과 — 5문장 요약')
p('네 상의 위치와 경계를 일관된 마스크로 제공하여 반복적인 영상 판독을 보조할 수 있다. 원본·예측·정답·오류를 함께 보여 주어 어떤 상과 영상에서 수정이 필요한지 검토할 수 있다. 영상별 지표와 원본 코드·가중치 해시를 연결해 실험 결과를 추적할 수 있다. 향후 독립 시편 평가와 추가 정답 검증이 확보되면 상 면적 분석 등의 후속 작업에 연결할 수 있다. 본 결과는 물성·탄성 측정의 대체 성능이나 현장 적용 효과를 검증한 것은 아니다.')
p('본 문서는 원본 양식의 필수 목차를 유지한 확장본이다. 사용자 요청에 따라 원본 양식의 10페이지 제한은 적용하지 않았다. 그림·표와 검증 20장 전체 비교를 포함하며, 미확인 인적 사항과 현재 모델의 test10 제출물 상태를 명시한다.')
B.append(dict(type='pagebreak'))
h('보고서 구성 및 핵심 결과',level=1)
p('① 분석(EDA) 개요: 분석 핵심 내용 / 분석 배경 / 분석 방법 및 분석 내용. ② 데이터 전처리: 전처리 방법 / 전처리 절차. ③ 모델링: 배경·알고리즘 / 모델링 방법 / 성능. 추가 장: 재현 정보, 한계와 추가 연구, 출처, 검증 20장 비교 부록.')
p(f'**핵심 결과는 stage1 R+E의 {pct(score("stage1"))}이다.** 같은 실행의 R {pct(score("R"))}보다 {delta("R","stage1"):+.6f}%p 높았다. F를 추가한 stage2는 {pct(score("stage2"))}로 stage1 대비 {delta("stage1","stage2"):+.6f}%p 높았지만, GitHub의 대표 후보 정책을 유지하여 stage1을 주 결과로 기록했다. 이 정책과 이번 단일 재학습에서 관측한 점수 순위는 서로 다른 정보다. [S03, S07, S08]')
p('보고서의 모든 성능 값은 image mIoU이며, 차이는 퍼센트포인트(%p)로 표시했다. GitHub 고정 가중치의 공개 기록, 프로젝트 과거 실험, 현재 새 가중치의 결과를 구분한다. 저장 마스크를 CPU에서 재채점했으며 새로운 모델 학습이나 GPU 추론은 수행하지 않았다. [S07]')
fig('11_stage_scores')

h('① 분석(EDA) 개요',level=1,newpage=True)
h('1-1. 분석 핵심 내용')
p('분석 대상은 현미경 영상의 픽셀 단위 다중 클래스 분할이다. 각 픽셀을 네 상 중 하나에 배정하고, 원본 해상도의 마스크로 복원한다. 데이터 수·해상도·밝기·클래스 구성·시편 식별자를 함께 확인하여 모델 입력과 평가 방식의 근거를 마련했다. 파일 이름의 A번호는 시편 그룹을 나타내는 식별자로 사용하되, 제공 자료만으로 촬영 조건이나 물리적 배율을 추정하지 않았다. [S02]')
tab('dataset','분석에 사용한 데이터',['분할','이미지','시편 ID','전체 픽셀','정답','사용 목적'],[
 [s,EDA['summary'][s]['images'],len(EDA['summary'][s]['specimens']),f'{EDA["summary"][s]["pixels"]:,}','없음' if s=='test' else '있음',{'train':'학습 및 내부 OOF','valid':'고정 모델 평가·진단','test':'입력 EDA; 현재 모델 성능 미평가'}[s]] for s in ['train','valid','test']],
 '원본 파일 해시를 기록하고 이전 데이터 분석의 입력 해시와 일치하는 것을 확인했다. [S02]')
fig('01_dataset')
h('1-2. 분석 배경')
p('단일 영상 안에서도 넓은 Al 영역과 상대적으로 작은 상이 함께 존재한다. 단순 픽셀 빈도 중심의 판단만으로는 작은 상의 오류를 충분히 설명하기 어렵다. 이 프로젝트의 평가식은 이미지와 클래스에 동일한 가중치를 주므로, 데이터의 픽셀 구성비와 목표 지표를 구분해야 한다. 작은 클래스의 오검출·미검출, 얇은 경계와 국소 조직의 혼동을 이미지별로 확인하는 것이 필요하다. [S01, S02]')
p('현재 접근은 하나의 큰 모델로 모든 경우를 해결하기보다 입력 표현과 특징 추출 방식이 다른 모델을 조합한다. 다만 모델 종류가 많다는 사실만으로 개선이 입증되는 것은 아니다. 보고서에서는 실제 같은 실행의 R, E, stage1, stage2를 함께 비교하고, 평균뿐 아니라 개별 영상의 손실도 제시한다. [S07]')
h('1-3. 분석 방법 및 분석 내용')
h('1-3-1. 해상도와 밝기 분포',level=3)
dims=sorted({(r['width'],r['height']) for r in EDA['records']},key=lambda d:d[0]*d[1])
tab('resolution','원본 해상도별 이미지 수',['너비 × 높이 (pixel)','train','valid','test'],[[f'{w} × {hh}',*[sum(r['split']==s and r['width']==w and r['height']==hh for r in EDA['records']) for s in ['train','valid','test']]] for w,hh in dims],
 '원본 해상도는 디지털 이미지 크기이며 촬영 배율이나 μm/pixel 정보가 아니다. [S02]')
fig('02_resolution')
tab('brightness','분할별 영상 밝기 통계',['분할','평균 밝기의 이미지 평균','밝기 표준편차의 이미지 평균'],[[s,f'{EDA["summary"][s]["mean"]:.3f}',f'{EDA["summary"][s]["std"]:.3f}'] for s in ['train','valid','test']],
 'PIL grayscale 0–255 기준. 각 영상을 동일하게 가중하여 평균했다. [S02]')
fig('04_brightness')
p('test의 평균 밝기가 train·valid보다 높게 관측되었지만, 10장의 입력 통계만으로 분포 이동의 원인이나 성능 변화를 확정할 수는 없다. 모델별 밝기 표준화, z-score/CDF 표현, width normalization은 이러한 차이를 처리하기 위한 설계이며, 본 EDA 자체가 각 전처리의 인과적 성능 개선 실험은 아니다. [S02]')
h('1-3-2. 클래스 구성 및 absent class',level=3)
tab('palette','클래스 ID와 정답 팔레트',['route ID','클래스','RGB','의미'],[[c,n,str(tuple(int(x) for x in PALETTE[c])),['Al₃Ni 상','Primary Si 상','Al 상','Eutectic Si 상'][c]] for c,n in enumerate(CLASSES)],
 'B4/HRNet 팀 순서는 [Al, Al3Ni, Primary Si, Eutectic Si]이며 route 순서와 다르다. 변환을 생략하면 평가가 잘못된다. [S04, S05]')
tab('class_distribution','클래스별 데이터 구성',['클래스','train 픽셀 %','valid 픽셀 %','train 등장/70','valid 등장/20','valid 평균 면적 %'],[[n,f'{EDA["summary"]["train"]["pixel_share"][c]*100:.4f}',f'{EDA["summary"]["valid"]["pixel_share"][c]*100:.4f}',EDA['summary']['train']['present'][c],EDA['summary']['valid']['present'][c],f'{EDA["summary"]["valid"]["image_share"][c]*100:.4f}'] for c,n in enumerate(CLASSES)],
 '픽셀 구성비는 split 전체 픽셀 기준, 평균 면적은 영상별 비율의 평균이다. 성능 지표가 아니다. [S02]')
fig('03_classes')
p('Primary Si는 train의 A57_step1_sample3, valid의 A71_step2_sample1 정답에 없다. 정답과 예측에 모두 없는 클래스의 IoU도 0이므로, 해당 이미지에서 다른 세 클래스를 완벽히 맞혀도 image mIoU의 최대값은 75%다. 이 규칙은 오류를 면제하거나 분모의 클래스 수를 3으로 줄이는 방식과 다르며 모든 실험에 동일하게 적용했다. [S01, S02, S07]')
h('1-3-3. 시편 단위 중복과 평가 범위',level=3)
tab('overlap','공식 분할 사이 공통 시편 ID',['분할 쌍','공유 개수','ID'],[[k,len(v),', '.join(v)] for k,v in EDA['overlaps'].items()],
 '파일명 A번호에 따른 집계. 제공된 train/valid 분할을 변경하지 않았다. [S02]')
fig('05_overlap')
p('train/valid가 공유하는 7개 시편 ID가 있으므로 valid20은 완전히 새로운 시편만의 외부 검증이 아니다. 내부 OOF는 train70 안에서 시편을 분리하여 보정기 입력의 자기 학습을 방지하지만, 공식 train/valid의 시편 중복이나 과거 후보 탐색의 선택 편향까지 제거하지 않는다. [S02, S13]')

h('② 데이터 전처리',level=1,newpage=True)
h('2-1. 전처리 방법')
p('원본 이미지와 마스크의 크기 일치, 이미지·정답 이름 대응, 팔레트 값과 파일 해시를 검증했다. 학습 전에는 분할별 파일 수를 고정하고, unknown mask color가 있으면 실행을 중단하도록 구성했다. 정답은 route ID 0–3으로 복원하며 원본 해상도의 좌표를 유지해 최종 채점한다. [S02, S03]')
tab('preprocessing','모델별 입력·공간 전처리',['경로','입력 표현','크기/통계','적용 이유와 주의'],[
 ['R 고해상도','원본 RGB + 밝기 표준화','768 tile, stride 576','학습 평균·분산과 원본의 style 혼합'],
 ['R B4 322','grayscale 3채널','3배 확대, crop 1536','ImageNet encoder 입력 형식 유지'],
 ['R B4 326','raw/z-score/CDF','3배 확대, 전체 영상 통계','상대 밝기·밝기 순위 표현'],
 ['R B4 329 / E B4','raw/z-score/CDF','width 4000, crop 1536','영상 너비 기준 배율 통일'],
 ['R native 269','전체 영상 표현 통계','scale 1, patch 1024','원본 해상도 경로 포함'],
 ['R HRNet','RGB 정규화','scale 2, patch 512','고해상도 특징 경로'],
 ['E HRNet','RGB 정규화','fold별 modal width 정규화 + scale 2','저장된 target_width metadata 사용'],
 ['F','E log probability + gray','full-image gray z-score, crop 256','추론도 같은 전체 영상 통계 사용']],'실제 실행 소스와 학습 args 기준이다. [S04–S06, S12]')
fig('06_representation')
p('B4의 세 입력 채널은 grayscale g∈[0,1]에서 raw, 전체 영상의 평균·표준편차를 이용한 z-score, 해당 영상의 CDF로 구성한다. patch를 자른 뒤 통계를 다시 계산하는 방식과 구분한다. 다음 식은 B4 표현의 scaling을 설명하며, ResNet 표현 계열의 CDF 구현을 동일 함수로 대체하는 근거는 아니다. [S05]')
eq('X = [(g−0.5)/0.25, clip((g−μ)/(σ+10⁻⁶), −4, 4)/3, (CDF(g)−0.5)/0.288675]',
 r'$$X=\left[\frac{g-0.5}{0.25},\;\frac{\mathrm{clip}((g-\mu)/(\sigma+10^{-6}),-4,4)}{3},\;\frac{\mathrm{CDF}(g)-0.5}{0.288675}\right].$$')
p('E HRNet은 각 fold의 학습 영상에서 결정한 modal target width를 checkpoint에 저장하고, 추론 시 원본 너비에 맞춘 effective input scale과 overlap을 계산한다. 현재 저장된 seed1/fold1의 target_width는 1230이다. 모든 fold 값을 임의로 하나로 덮어쓰지 않고 각 checkpoint의 전처리 metadata를 읽는다. [S05]')
tab('augmentation','주요 학습 증강',['모델','공간 증강','밝기·표현 증강','실행상 구분'],[
 ['B4 전문가','random crop 1536, D4','brightness/contrast, gamma, blur/noise','증강과 추론 D4 TTA는 별개'],
 ['HRNet 전문가','random crop, flip/90° rotation, deformation','brightness/contrast, gamma, blur','seed1/2, rng_mode=legacy'],
 ['F 보정기','256 crop, 좌우/상하 반전, 90° 회전','전체 영상 gray normalization 유지','클래스 0/3 위치를 고려한 혼합 crop'],
 ['R 기존 계열','원래 각 학습 소스의 crop·증강','기존 학습 레시피 유지','현재 실행 plan과 source snapshot이 기준']],'광학 배율·물리 길이를 새로 합성하거나 정답 정보를 추론 입력에 넣지 않았다. [S05, S06, S12]')
h('2-2. 전처리 절차')
p('절차 1: 원본 train70/valid20과 test10 입력을 식별하고 해시·크기·색상을 확인한다. 절차 2: train70을 A번호 기준 4-fold로 나누고, 각 fold의 학습·held-out 시편이 겹치지 않는지 확인한다. 절차 3: 각 모델에 맞는 입력 표현과 크기 정규화를 적용한 뒤 학습 patch를 추출한다. 절차 4: 학습된 모델의 예측을 원본 좌표로 복원하고 클래스 순서를 통일한다. 절차 5: family 확률의 반올림 순서를 유지하여 E를 결합하고, train70 OOF를 F의 입력으로 저장한다. 절차 6: valid20의 최종 label을 PNG로 저장한 뒤 정답과 독립 채점한다. [S03–S07, S13]')
fig('09_oof')
p('E의 valid 예측은 각 family의 네 fold를 float32로 평균하고, family별 평균을 한 번 float16으로 반올림한 다음 다시 float32로 결합한다. 각 fold를 먼저 float16으로 바꾸어 평균하는 연산과 일반적으로 같지 않다. 반면 OOF에서는 해당 이미지를 제외한 한 fold의 모델만 각 family에서 사용한다. 이 수치 처리와 클래스 순서가 재현의 일부다. [S05]')

h('③ 모델링',level=1,newpage=True)
h('3-1. 배경/알고리즘 소개')
p('U-Net은 encoder에서 얻은 문맥과 decoder의 위치 복원을 skip 연결로 결합한다. U-Net++는 중첩된 skip 경로를 사용하여 여러 깊이의 특징을 연결한다. 이 프로젝트의 ResNet compact U-Net++ 구현과 SMP의 HRNet U-Net++는 서로 다른 구현이므로 같은 이름만으로 구조가 완전히 같다고 설명하지 않는다. [P01, P02, S04, S05]')
p('ResNet은 잔차 연결을 통해 특징을 학습하고, EfficientNet은 네트워크의 깊이·너비·입력 크기를 함께 고려한 계열이다. HRNet은 여러 해상도의 특징 경로를 유지하고 정보를 교환하는 구조다. 여기서는 이 서로 다른 특징 추출 계열을 분할 decoder와 연결했다. 원 논문의 벤치마크 성능을 본 합금 영상의 성능으로 이전해 주장하지 않는다. [P03, P04, P05]')
p('ASPP는 서로 다른 dilation을 가진 convolution으로 여러 범위의 문맥을 처리한다. 프로젝트의 ResNet 분기에는 잔차 형태의 ASPP가 들어가며 일부 경로는 원본 해상도의 native stem을 추가한다. TENT는 정답 없이 예측 entropy를 줄이는 test-time adaptation 방법으로, R의 선택된 고해상도 경로에서 이미지마다 적용한다. 고정 checkpoint 학습과 추론 중 일시적 적응을 구분해야 한다. [P06, P07, S04]')
fig('08_architecture')
p('Model349 또는 fusion805라는 이름은 단일 신경망 층의 이름이 아니라 모델 집합, 입력 표현, 확률 결합, 라우팅, 후처리를 묶은 실험·패키지 식별자다. fusion805의 파일명 역시 현재 새 가중치가 80.5%를 보장한다는 뜻이 아니다. GitHub에서는 F를 제외한 stage1을 대표 후보로 두고 stage2를 탐색 기록으로 보존한다. [S08]')

h('3-2. 모델링 방법')
fig('07_pipeline')
h('3-2-1. R: 14개 checkpoint의 exh14 경로',level=3)
tab('release_models','R 구성 모델과 학습 계보',['구성','개수','초기값/부모','학습량','seed'],[
 ['Model109 ASPP',1,'보존 Model98','4 epochs, patch512','42'],['restart51',1,'새 Model109','6 epochs, patch768','123'],
 ['restart59',1,'새 Model109','5 epochs, patch768','456'],['restart61 pairwise',1,'새 restart51','4 epochs, patch768','42'],
 ['Model269 native',1,'보존 Model257','4 epochs, patch1024','269'],['B4 Model322',2,'ImageNet; folds3·4','각 16,000 iterations','각 0'],
 ['B4 Model326',2,'ImageNet; folds1·2','각 16,000 iterations','각 0'],['B4 Model329',3,'ImageNet; folds1·3·4','각 16,000 iterations','각 0'],
 ['HRNet336',1,'ImageNet','70 epochs','0'],['HRNet347',1,'새 HRNet336','12 epochs','347'],['합계',14,'Model98·257 이전 계보는 범위 밖','각 구성 1회 재학습','—']],
 '기존 완성된 R 가중치를 이번 출력으로 재사용하지 않았다. 단, 보존된 선행 Model98·257과 ImageNet 사전학습 초기값은 사용했다. [S03, S12]')
p('고해상도 경로는 Model109·restart51·restart59·restart61 중 영상 밝기 규칙에 맞는 checkpoint를 선택한다. EMA 분기를 제거한 lite 설정에서는 과거 EMA 경로 대신 restart51을 재사용한다. 선택된 모델에 대해 TENT 5회, learning rate 10⁻⁵를 적용하고, 원본 및 밝기 표준화 영상의 768 tile(stride576) D4 예측을 섞는다. 1024 tile의 추가 고해상도 경로와 별도 pair 확률 혼합은 비활성화되어 있다. 저장된 원본 checkpoint를 덮어쓰지 않는다. [S04]')
p('R의 B4 구성은 322의 fold3·4, 326의 fold1·2, 329의 fold1·3·4만 사용한다. 따라서 과거 전체 Model349의 B4 12개와 현재 exh14의 B4 7개를 혼동하면 안 된다. Native269 1개와 HRNet336·347 2개가 추가되어 총 14개다. [S04, S12]')
tab('r_fusion','R의 고정 클래스별 결합 설정',['결합 단계','Al3Ni','Primary Si','Al','Eutectic Si'],[
 ['base 전문가/source','326: γ0.7','326: γ1.0','322: γ0.6','269: γ0.7'],['후속 Model329 γ','0.5','1.0','0.0','0.0'],
 ['후속 HRNet336 γ','0.0','0.25','0.75','0.75'],['후속 HRNet347 γ','0.0','0.0','0.25','0.0']],
 '순차 결합 설정이며 독립 성능 기여율이나 모든 네트워크에 대한 단일 전역 가중치가 아니다. 설정: frozen/configs/model349_lite.yaml. [S04]')
p('R의 prior_scale=1.25, beta_scale=0.8, Eutectic delta=0.15, Gaussian blend=0.5, Eutectic edge blend=0.25, Primary Si cleanup threshold=128을 유지했다. 밝기 prior, 클래스 bias, 공간 보정과 연결 성분 규칙은 학습된 네트워크 출력 뒤의 고정 처리다. 이 설정을 이번 valid20 점수에 맞춰 재탐색하지 않았다. [S04]')
tab('tta','실제로 실행된 추론 증강',['경로','실제 실행','주의'],[
 ['R 고해상도 base','D4 + TENT5, 768 tile','원본/스타일 영상 결합'],['R B4','tta=False','YAML의 b4_tta 이름과 구분'],
 ['R HRNet','tta=False','tta_mode 문자열이 있어도 비활성'],['R native269','tta_mode=none','학습 중 선택 검증의 D4와 다름'],
 ['E B4','D4','4-fold family 평균'],['E HRNet','flip4','seed별 4-fold 평균 + width normalization'],['F','단일 전체 이미지 잔차 예측','F가 Al인 위치만 stage2에 반영']],
 '문서화되지 않은 추론을 추정하지 않고 infer_lite.py의 실제 호출 인자를 기준으로 기록했다. [S04, S05]')
h('3-2-2. E: B4 및 HRNet 전문가 12개',level=3)
tab('expert_training','E 학습 설정',['항목','B4','HRNet'],[
 ['구조','EfficientNet-B4 + U-Net','HRNet-W48 + U-Net++ + boundary head'],['구성','4 folds × seed0','4 folds × seeds1·2'],
 ['초기화','ImageNet encoder','ImageNet encoder'],['학습량','16,000 iterations/fold','70 epochs/fold'],['batch / accumulation','2 / 1','4 / 2'],
 ['crop','1536','512'],['learning rate','3e-4','3e-4'],['weight decay','1e-2','1e-4'],['optimizer','AdamW','AdamW'],
 ['손실','CE + Dice','0.5 CE + 0.5 Dice + boundary loss'],['checkpoint 선택','last.pt','last.pt'],
 ['valid20 학습 중 평가','no_valid_eval','val_every=0'],['worker/RNG','고정 worker seed','rng_mode=legacy, seeds1·2']],
 'HRNet boundary loss는 BCE+Dice이며 가중치 1.0으로 더한다. CE 클래스 가중치는 E에서 사용하지 않는다. [S05, S12]')
tab('folds','E의 시편 단위 4-fold',['fold','학습 이미지','held-out 이미지'],[[f,52 if f<3 else 53,18 if f<3 else 17] for f in range(1,5)],
 'train70의 59개 시편 그룹을 고정 분할. 각 학습 영상은 정확히 한 held-out fold에 들어간다. [S12, S13]')
eq('P_E = 0.50 · Q(P_B4) + 0.25 · Q(P_HR,seed1) + 0.25 · Q(P_HR,seed2)',
 r'$$P_E=0.50Q(\bar P_{B4})+0.25Q(\bar P_{HR,1})+0.25Q(\bar P_{HR,2}),$$')
p('여기서 각 P̄는 family의 네 fold 평균이고 Q는 float16 반올림 후 float32 복원이다. OOF에서는 평균 대신 해당 held-out fold 모델의 확률을 사용한다. 팀 순서를 route 순서로 [1,2,0,3] 재배열하며, argmax 동률에서는 원래 팀 클래스 우선순위를 유지한다. [S05]')
h('3-2-3. F: OOF 확률의 제한된 잔차 보정',level=3)
p('F는 E의 예측을 입력으로 사용하는 3층 convolution 모델이다. 네 채널 log probability와 한 채널 gray로부터 5채널 입력을 만든다. 구조는 Conv3×3(5→32)–SiLU–Conv3×3(32→32)–SiLU–Conv3×3(32→4)이며 총 11,876개 parameter다. 마지막 convolution을 0으로 초기화하므로 시작 시 출력은 부모 log probability와 같다. [S06]')
eq('L = log(max(P_E, 10⁻⁷)), Δ = tanh(Fθ([clip(L−mean_c(L),−8,8)/4, 2g_z−1])), L′ = L+Δ',
 r'$$L=\log(\max(P_E,10^{-7})),\quad \Delta=\tanh F_\theta\!\left([\mathrm{clip}(L-\mathrm{mean}_c L,-8,8)/4,\;2g_z-1]\right),\quad L\prime=L+\Delta.$$')
p('gray 입력 g_z는 원본 전체 영상의 z-score를 ±3으로 자르고 z/6+0.5로 [0,1] 범위에 맞춘다. 표준편차가 10⁻⁶보다 작으면 0.5를 사용한다. 학습에서는 이 변환 후 crop하고, 추론에서도 전체 이미지 통계를 사용한다. tanh에 의해 각 logit 잔차는 −1부터 1 사이로 제한된다. [S06]')
eq('Loss_F = 0.5 CE + 0.5 Dice loss + 0.05 mean(Δ²)',r'$$\mathcal L_F=0.5\mathcal L_{CE}+0.5\mathcal L_{Dice}+0.05\,\mathrm{mean}(\Delta^2).$$')
p('학습 데이터는 새 E 모델의 train70 OOF 확률, 원본 gray, train70 정답이다. 이미지당 16개 sample, batch8로 epoch당 140 step을 수행하며 8 epochs의 총 1,120 update를 고정한다. seed81349, AdamW lr=10⁻³, weight decay=10⁻⁴, cosine T_max=8, gradient clip=1을 사용했다. 클래스 0·3 중심 crop과 균등 crop을 섞으며, valid/test 정답은 F 학습에 넣지 않았다. [S06, S13]')
fig('19_refiner_loss')
p(f'실제 F 학습 receipt에는 8 epochs, 1,120 optimizer updates, {T["refiner"]["elapsed_seconds"]:.3f}초가 기록되어 있다. 최종 epoch8을 사용했으며 validation 점수가 높은 epoch를 고르는 탐색은 하지 않았다. 이 손실 곡선은 학습 목적함수의 변화로, 일반화 성능의 향상 곡선을 의미하지 않는다. [S06, S10]')
h('3-2-4. 두 단계 label 결합',level=3)
eq('ŷ₁(x) = ŷ_E(x) if ŷ_E(x) ∈ {Al3Ni, Al}; otherwise ŷ_R(x)',r'$$\hat y_1(x)=\begin{cases}\hat y_E(x),&\hat y_E(x)\in\{\mathrm{Al3Ni},\mathrm{Al}\},\\\hat y_R(x),&\text{otherwise}.\end{cases}$$')
eq('ŷ₂(x) = Al if ŷ_F(x) = Al; otherwise ŷ₁(x)',r'$$\hat y_2(x)=\begin{cases}\mathrm{Al},&\hat y_F(x)=\mathrm{Al},\\\hat y_1(x),&\text{otherwise}.\end{cases}$$')
p('stage1은 E의 모든 클래스 출력을 통째로 대체하지 않는다. E가 Al₃Ni 또는 Al로 판단한 위치에만 E label을 적용한다. stage2도 F가 Al로 판단한 위치만 덮어쓴다. 신뢰도 threshold나 이번 validation 점수에 따른 추가 규칙은 사용하지 않았다. [S08, S03]')

h('3-3. 성능(필수)',newpage=True)
h('3-3-1. 공식 지표와 독립 재채점',level=3)
eq('IoUᵢ,c = TPᵢ,c / (TPᵢ,c + FPᵢ,c + FNᵢ,c), 단 union=0이면 0',r'$$\mathrm{IoU}_{i,c}=\begin{cases}\frac{TP_{i,c}}{TP_{i,c}+FP_{i,c}+FN_{i,c}},&TP_{i,c}+FP_{i,c}+FN_{i,c}>0,\\0,&\text{otherwise}.\end{cases}$$')
eq('image mIoU = (1/20) Σᵢ₌₁²⁰ [(1/4) Σc₌₁⁴ IoUᵢ,c]',r'$$\mathrm{image\ mIoU}=\frac{1}{20}\sum_{i=1}^{20}\left(\frac14\sum_{c=1}^{4}\mathrm{IoU}_{i,c}\right).$$')
p('각 이미지에서 네 클래스 IoU를 먼저 평균하고 그 값을 20장에 대해 평균한다. 정답과 예측에 없는 클래스도 0으로 포함하며, 전체 영상을 하나로 합친 global mIoU는 계산하지 않았다. 100%−image mIoU는 지표의 보수값이며 오분류 픽셀 비율이 아니다. [S01, S07]')
fig('10_metric')
p('R·stage1·stage2의 저장 label PNG 60장을 정답과 다시 대조하여 총 240개 클래스 IoU와 3개 최종 평균이 기존 JSON과 10⁻¹² 이내에서 일치함을 확인했다. E 단독 점수는 저장된 E 확률에서 원래 동률 우선순위를 유지한 argmax로 새로 계산했다. 이는 추가 학습이나 모델 재추론이 아닌 저장 출력의 평가다. [S07]')
tab('main_scores','현재 실행의 valid20 성능',['구성','image mIoU','100%−mIoU','R 대비 %p','용도'],[[{'R':'R (exh14)','E':'E 단독','stage1':'Stage1 R+E','stage2':'Stage2 R+E+F'}[k],pct(score(k)),pct(1-score(k)),f'{(score(k)-score("R"))*100:+.6f}',{'R':'동일 실행 기준','E':'저장 확률 진단','stage1':'대표 결과','stage2':'보정기 비교'}[k]] for k in ['R','E','stage1','stage2']],
 '모두 같은 valid20에 대한 현재 새 가중치의 결과다. 별도 직전 R 실험의 80.052074%와 현재 R 80.051924%를 구분한다. [S03, S07]')
p(f'E 단독은 {pct(score("E"))}로 R보다 낮지만, 제한된 클래스 선택 결합은 R보다 {delta("R","stage1"):+.6f}%p 높았다. 따라서 이 결과는 E 전체로 교체하는 것과 E의 일부 결정을 활용하는 것이 다름을 보여 준다. 이 관측만으로 원인을 특정 architecture나 전처리 하나에 귀속시키지는 않는다. [S07]')
h('3-3-2. 클래스별 및 이미지별 결과',level=3)
tab('class_iou','이미지별 class IoU의 20장 평균',['클래스','R','E','Stage1','Stage2','Stage1−R %p'],[[n,*[pct(M[k]['class_iou'][c],4) for k in ['R','E','stage1','stage2']],f'{(M["stage1"]["class_iou"][c]-M["R"]["class_iou"][c])*100:+.6f}'] for c,n in enumerate(CLASSES)],
 'class IoU도 이미지 단위 평균이다. 정답 없는 Primary Si 영상 1장의 0을 포함했다. [S07]')
fig('13_class_iou')
p('stage1에서 Al₃Ni, Primary Si, Eutectic Si의 평균 IoU는 R보다 높고 Al은 소폭 낮았다. 최종 평균은 이 네 변화를 같은 비중으로 합친 값이다. E의 Al₃Ni/Al label만 덮어쓰더라도 다른 클래스의 FP/FN이 달라지므로 나머지 클래스 IoU도 변할 수 있다. Eutectic Si의 IoU가 가장 낮아 남은 오류의 주요 점검 대상으로 볼 수 있지만, 클래스별 원인과 개선 방법을 이 표만으로 확정할 수는 없다. [S07]')
names=[r['image'] for r in M['stage1']['images']]
tab('per_image','검증 20장 단계별 점수',['이미지','R %','Stage1 %','Stage2 %','S1−R %p','S2−S1 %p'],[[n,*[f'{img(k,n)["image_miou"]*100:.6f}' for k in ['R','stage1','stage2']],f'{(img("stage1",n)["image_miou"]-img("R",n)["image_miou"])*100:+.6f}',f'{(img("stage2",n)["image_miou"]-img("stage1",n)["image_miou"])*100:+.6f}'] for n in names],
 '원본 순서에 맞춰 이름으로 대응시켰다. 비교 이미지는 부록 A에 20장 모두 제공한다. [S07]')
fig('14_per_image');fig('15_heatmap')
tab('stage1_per_class','Stage1 이미지별 4클래스 IoU',['이미지','Al3Ni %','Primary Si %','Al %','Eutectic Si %','image mIoU %'],[[n,*[f'{v*100:.4f}' for v in img('stage1',n)['class_iou']],f'{img("stage1",n)["image_miou"]*100:.6f}'] for n in names],'Stage2의 전체 클래스별 값은 evidence/audited_metrics.json에 같은 구조로 포함했다. [S07]')
h('3-3-3. 평균 개선과 개별 영상의 손실',level=3)
tab('transition_summary','단계별 개선·악화와 변경 픽셀',['전환','개선/악화 영상','변경 픽셀','오류→정답','정답→오류','오류→다른 오류'],[[k,f'{v["improved"]} / {v["declined"]}',f'{v["changed"]:,}',f'{v["corrected"]:,}',f'{v["worsened"]:,}',f'{v["wrong_to_wrong"]:,}'] for k,v in C['summary'].items()],
 '진단용 픽셀 수이며 image mIoU의 정의를 대체하지 않는다. 동일 픽셀을 각 전환에서 독립 비교했다. [S07]')
fig('16_deltas');fig('17_pixel_changes')
p('stage1은 16개 이미지에서 개선되고 4개에서 악화됐다. 최대 개선은 A11_step2_sample4의 +0.624725%p, 최대 악화는 A153_step3_sample3의 −0.541998%p다. 변경된 177,703픽셀 중 83,508픽셀은 정답으로 교정됐고 82,597픽셀은 새 오류가 됐다. 순 교정 수가 작더라도 이미지·클래스별 동일 가중 지표에서는 분포에 따라 평균 변화가 달라질 수 있다. [S07]')
p('F를 추가하면 14개 이미지에서 개선되고 6개에서 악화됐다. 최대 개선은 A63_step1_sample4의 +0.231528%p, 최대 악화는 A72_step4_sample4의 −0.020728%p다. 변경 픽셀은 14,878개로 stage1 변경보다 적다. 국소 수정의 평균 이득은 관측되지만, 단일 seed·단일 validation 결과이므로 안정적 일반화 이득으로 확대 해석하지 않는다. [S07]')
h('3-3-4. 정성적 오류 사례',level=3)
a71=img('stage1','A71_step2_sample1');a44=img('stage1','A44_step4_sample3')
p(f'A71_step2_sample1은 stage1 {pct(a71["image_miou"])}로 최저 점수다. Primary Si의 IoU=0과 함께 나머지 클래스 IoU도 완벽하지 않다. 클래스별 IoU는 Al₃Ni {pct(a71["class_iou"][0],3)}, Al {pct(a71["class_iou"][2],3)}, Eutectic Si {pct(a71["class_iou"][3],3)}다. absent class 규칙만으로 모든 오차를 설명하거나 이 영상을 평가에서 제외하지 않는다. [S07]')
fig('20_hard_case')
p(f'A44_step4_sample3은 stage1 {pct(a44["image_miou"])}로 가장 높다. 그림의 오류 맵은 맞은 픽셀을 흰색, 틀린 픽셀을 그 위치의 정답 색으로 표시하여 어떤 상이 다른 상으로 분류됐는지 보여 준다. 100%−image mIoU와 오류 맵의 색 픽셀 비율은 같은 값이 아니다. [S07]')
fig('21_best_case');fig('22_refiner_gain')
if '23_stage1_loss' in F:fig('23_stage1_loss')
h('3-3-5. GitHub 공개 결과와 재학습 결과의 비교',level=3)
tab('github_comparison','출처·가중치·실행 유형별 점수',['출처/실행','구성','valid20 image mIoU','해석'],[
 ['GitHub canonical','R exh14','80.231231%','공개 고정 가중치 기록'],['GitHub 저장 부모 출력','R+E','80.4746%','원문 반올림; 대표 후보 선택 기록'],
 ['GitHub 저장 부모 출력','R+E+F','80.500689%','F 탐색 보존 결과'],['GitHub fresh 원본 재추론','R','80.230986%','같은 fresh 실행의 기준'],
 ['GitHub fresh 원본 재추론','R+E','80.474424%','고정 가중치를 원본 영상부터 추론'],['현재 로컬 재학습','R+E',pct(score('stage1')),'새 가중치, 주 결과'],
 ['현재 로컬 재학습','R+E+F',pct(score('stage2')),'새 가중치, 추가 비교']],
 'GitHub 값은 원문 기록이며 이 보고서에서 해당 원본 가중치를 다시 실행한 수치가 아니다. [S03, S08, S09]')
fig('12_provenance_scores')
p('GitHub stage1의 시편 제외 선택 검증 80.4088%는 매번 대상 시편을 제외하고 69개 실행 × 30개 규칙 전체를 다시 선택한 결과다. E 하나를 바깥에서 고정한 평가로 설명하면 부정확하다. 다만 후보 모델과 R 자체의 과거 valid 기반 개발 선택 편향까지 제거한 완전한 외부 평가도 아니다. [S08, S09]')
p('GitHub에서 F가 포함된 stage2는 전체 valid로 선택한 stage1을 고정한 조건부 검증에서 약 80.3160%로 stage1 대비 손실이 관측되어 미채택됐다. 따라서 현재 단일 재학습에서 stage2가 조금 높다는 사실과, 배포 문서의 대표 모델이 stage1이라는 사실을 모두 기록했다. 현재 새 가중치에 대해 같은 시편 제외 선택 검증을 다시 수행한 것은 아니다. [S08]')
h('3-3-6. 프로젝트 내 관련 실험의 맥락',level=3)
history=[('0930_1','Model349 로컬 학습·조합','80.068799%','기존 전체 경로; 중간 선택 포함'),('1002_1','Model349 CE 클래스 가중치 제거','80.040537%','다른 학습 조건'),('1006_1','Model349 재실행','80.080398%','별도 재학습 실행'),('1006_2','Model397 전문가 검증','79.430160%','OOF 번들에 추가한 로컬 valid 평가'),('1006_4','Model364 재개 결과','79.915894%','별도 구성; overlap 수정'),('20261008_1','Model349 100개 학습 시드 탐색','80.255658%','동일 valid에서 후보 선택'),('20261009_1','exh14 14개 재학습','80.052074%','직전 독립 R 실행')]
tab('history','현재 프로젝트의 관련 결과',['결과 폴더 끝부분','실험','image mIoU','비교 시 구분'],history,
 '실제로 존재하는 각 결과 md의 기록. 서로 다른 선택 예산·모델·학습 실행이므로 통제된 ablation이나 동일 예산 순위표가 아니다. 전체 파일 목록은 project_result_inventory.csv. [S11]')
p('Model349 시드 탐색은 계획한 학습 100개를 완료하고 최종 형태로 평가한 후보 88개에서 조합을 선택한 기록이다. 현재 fusion805는 고정된 27개 구성의 한 번 재학습이므로 두 점수의 근접성이나 우열을 같은 탐색 예산의 비교로 해석하지 않는다. Model364의 중단 보고서(model_results_1006_3)는 최종 점수가 없으므로 완료 실험의 수치로 채우지 않았다. [S11]')
h('3-3-7. test10 제출 산출물의 상태',level=3)
p('원본 양식은 test10 예측 마스크 제출을 요구한다. 현재 model_results_20261009_2에는 새 가중치의 valid20 결과가 있으며, 이 실행에서 생성한 test10 결과 폴더는 확인되지 않았다. 따라서 이 보고서는 현재 재학습 모델의 test10 제출 마스크가 완성되었다고 기재하지 않는다. 제공된 test10 입력은 EDA에만 사용했고, 정답이 없으므로 test mIoU를 계산하지 않았다. [S01–S03]')
p('GitHub fresh 보고서의 test10은 별도 고정 가중치 실행이며 새 R 대비 일치율 98.21783315%가 기록되어 있다. 이는 두 모델 예측 사이의 일치율이지 정답 대비 정확도 또는 test mIoU가 아니다. 과거 다른 모델의 test 마스크를 현재 모델의 제출 결과로 대체하지 않는다. [S09]')

h('④ 재현 정보와 실행 비용',level=1,newpage=True)
env=read(PACKAGE/'evidence/local_preflight.json')
tab('environment','현재 학습·평가 환경',['항목','기록'],[['Conda','komap'],['Python 경로',env['python']],['GPU','NVIDIA GeForce RTX 4090 × 6 (물리 2,3,4,5,6,7)'],*[ [k,v] for k,v in env['versions'].items()],['학습 작업','R14 + E12 + F1 = 27'],['전체 작업','학습27 + 준비/OOF/valid16 = 43'],['전체 경과',f'{T["total_seconds"]/3600:.4f}시간 ({T["total_seconds"]:.3f}초)'],['최종 패키징·평가·저장',f'{T["finalization_seconds"]:.3f}초'],['GPU 작업 경과 합계',f'{T["gpu_job_seconds"]/3600:.4f} GPU·시간']],
 'GPU 작업 경과 합계는 subprocess 시작~종료 합이며 순수 GPU kernel 시간/에너지/가격이 아니다. 최종 단계 시간도 순수 추론 지연이 아니다. [S10, S12]')
fig('18_gpu_timeline')
p('작업 하나에 GPU 하나를 배정하고, 부모 checkpoint가 완성된 다음 자식 작업을 시작했다. 이는 하나의 모델을 6대에 나누어 학습하는 DDP 방식이 아니다. 실제 전체 단계 경과는 약 3시간이며, 모든 모델의 epoch 수를 합한 값이나 원본 GitHub 하드웨어에서의 시간으로 해석하지 않는다. [S10]')
p('실험 구축 시 GitHub commit d4ef4b08455a6bd849225f672fdcbaa23e24cdb1의 fusion805 ZIP을 다운로드하고 공개 checksum을 검증했다. 보고서 작성 시에는 ac6252bef323bdba7482f89a61420312310901e9의 최신 설명을 추가 확인했다. 두 commit의 역할을 구분하여 실행 당시 소스와 현재 설명 문서의 출처를 보존했다. [S03, S08, S09]')
tab('artifacts','산출물과 확인 방법',['항목','경로/역할'],[['본 보고서','모델링_결과서_Fusion805.doc / .docx / .md / .pdf'],['그림','figures/*.png (300dpi), *.svg, catalog.json'],['표','tables/*.md, *.csv; Word 문서에는 편집 가능한 표로 삽입'],['검증 20장 비교','appendix/stage1 및 appendix/stage2, 각 20개 PNG'],['수치·출처','evidence/audited_metrics.json, pixel_changes.json, GitHub 사본'],['분석·생성 소스','source/analyze.py, figures.py, report.py, render.py'],['실행 소스 보존','evidence/execution_source/'],['원본 모델 보존 위치','results/model_results_20261009_2/selected_model/'],['검증 기록','validation/; 문서 변환·이미지·링크·해시 점검']],
 '보고서 ZIP에 다중 GB 학습 가중치와 전체 train 원본 데이터는 중복 포함하지 않는다. 가중치 manifest와 기존 경로는 보존한다.')
p('현재 결과를 확인하는 기준 파일은 complete.json, fusion_comparison.json, metrics.json, selected_model의 model_manifest.json이다. 구성 가중치의 해시는 새 재학습 패키지 기준이며 원본 GitHub 가중치 해시와 같다고 주장하지 않는다. Word의 DOC는 실제 Word 97–2003 바이너리 형식으로 변환하고, 편집 호환성을 위해 DOCX와 열람용 PDF도 함께 제공한다.')

h('⑤ 한계와 추가 연구',level=1,newpage=True)
p('첫째, valid20은 개발 과정에서 반복적으로 사용되었고 train과 일부 시편 ID가 겹친다. 따라서 본 점수는 제공된 검증셋의 성능이며 새로운 시편·촬영 조건에 대한 독립 일반화 성능은 아니다. 둘째, 현재 실험은 구성별 고정 seed의 한 번 재학습이다. 소수점 여섯 자리 표시는 계산값을 추적하기 위한 정밀도이며 그만큼 작은 변화가 통계적으로 확실하다는 뜻은 아니다. [S02, S07, S08]')
p('셋째, GitHub 고정 모델과 현재 재학습 결과 사이에는 가중치·환경·학습 난수·선택 과정의 차이가 있다. 현재 R+E가 GitHub fresh R+E보다 낮은 차이를 특정 라이브러리나 normalization 변경 하나의 영향으로 단정할 수 없다. 넷째, GitHub의 F 조건부 검증 결과와 현재 단일 실행의 작은 평균 이득을 함께 고려해야 한다. [S08, S09]')
p('다섯째, 영상별 손실이 남는다. 현재 stage1의 최악 손실은 −0.541998%p로 GitHub의 영상별 손실 기준 −0.10%p를 만족하지 않는다. 단일 평균값만으로 기존 base를 대체할 근거가 충분하다고 하지 않는다. 여섯째, 현재 새 가중치의 test10 제출 마스크는 본 보고서 생성 범위에서 새로 만들지 않았고, test GT 부재로 성능을 검증할 수 없다. [S07–S09]')
tab('future','추가 연구와 확인할 질문',['우선순위','연구 내용','평가 설계','확인할 사항'],[
 ['1','독립 시편/촬영 배치 검증','시편 단위 외부 hold-out 또는 사전 고정 nested CV','현재 개선이 새로운 시편에서도 유지되는가'],
 ['2','R/E/F paired seed 반복','동일 seed·동일 split의 paired 비교, 평균과 분산 보고','+0.030761%p가 학습 변동보다 큰가'],
 ['3','Eutectic Si 및 A153 오류 검토','정답 품질·경계별 오류 검토, 사전 고정 ablation','부담이 큰 오류 유형은 무엇인가'],
 ['4','F 규칙의 안정성','선택셋/평가셋 분리; epoch·threshold 선택을 중첩 검증','특정 validation에 맞춘 보정인가'],
 ['5','모델 수·지연 최적화','점수와 영상별 손실을 함께 제한한 distillation/축소','26개/27개 checkpoint 비용을 줄일 수 있는가'],
 ['6','현재 모델 test10 산출','가중치·전처리 hash 고정 후 별도 사용자 실행','제출 파일 수·크기·팔레트·좌표가 맞는가']],
 '향후 제안이며 이번 보고서에서 수행하거나 효과를 확인한 실험이 아니다.')
p(f'현재 확인된 성과는 **고정된 27개 구성의 재학습을 완료하고, 같은 실행의 R 대비 stage1 {delta("R","stage1"):+.6f}%p를 얻은 것**이다. stage2의 추가 이득과 영상별 손실, 검증셋의 범위를 함께 공개하여 다음 실험의 기준으로 사용한다. 현재 로컬·GitHub 기록 모두를 근거로 81% 달성이나 현장 배포 준비 완료를 주장하지 않는다. [S03, S07–S09]')
h('⑥ 출처와 근거 자료',level=1,newpage=True)
p('S로 시작하는 인용은 프로젝트 자료 또는 고정 GitHub 문서, P로 시작하는 인용은 알고리즘의 원 논문이다. 모델 구현과 성능의 직접 근거는 실행 소스·결과 JSON이며, 원 논문은 알고리즘 배경 설명에 사용했다. 상대 경로는 본 보고서 폴더 기준이다.')
for key,row in refs.items():B.append(dict(type='reference',id=key,**row))
h('부록 안내',level=2)
p('부록 A는 validation 20장의 stage1·stage2 원본 비교 이미지 40개를 모두 포함한다. 각 페이지의 왼쪽은 대표 stage1, 오른쪽은 stage2이며, 각 이미지 내부는 기존 실험과 같은 2×2 배치다. 원본 PNG는 appendix/에 별도로 보존하므로 확대하여 확인할 수 있다. 표·그림의 개별 파일과 생성 코드는 ZIP에도 포함했다.')
B.append(dict(type='appendix_start'))
for i,n in enumerate(names,1):
    s1=img('stage1',n)['image_miou'];s2=img('stage2',n)['image_miou']
    B.append(dict(type='appendix_image',index=i,name=n,text=f'Stage1 image mIoU {pct(s1)} (100%−mIoU {pct(1-s1)})  |  Stage2 {pct(s2)} (100%−mIoU {pct(1-s2)})  |  Δ {(s2-s1)*100:+.6f}%p',files=[f'appendix/stage1/{n}.png',f'appendix/stage2/{n}.png']))

write(PACKAGE/'source/manuscript.json',dict(blocks=B,references=refs))
render(PACKAGE,B,refs)
print('Rendered matching Markdown and DOCX manuscript with all tables, figures and validation panels.')
