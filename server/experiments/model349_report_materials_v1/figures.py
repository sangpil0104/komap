"""Publication-sized PNG figures from read-only Model349 analysis."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import Normalize
from tqdm import tqdm

ROOT=Path('/Data3/sppark/komap')
HERE=Path(__file__).resolve().parent
ANALYSIS=HERE/'outputs/analysis_20261007_1'
RUN=ROOT/'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'
SOURCE=RUN/'source_snapshot'
PACKAGE=ROOT/'results/model349_report_materials_20261007_1'
FIG=PACKAGE/'figures'
FONT=Path('/tmp/model349_report_fonts/NanumGothic-Regular.ttf')
font_manager.fontManager.addfont(FONT)
FONTNAME=font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({'font.family':FONTNAME,'axes.unicode_minus':False,'font.size':11,
                     'axes.titlesize':14,'axes.labelsize':11,'figure.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
BLUE='#2455A4'; TEAL='#168D91'; ORANGE='#D88437'; INK='#20324A'; GRAY='#66758B'
SPLITS=['train','valid','test']; SCOLORS=[BLUE,TEAL,ORANGE]
NAMES=['Al3Ni','Primary Si','Al','Eutectic Si']
PCOLORS=['#194C99','#4CB24C','#997F4C','#A9A9A9']
EDA=json.loads((ANALYSIS/'eda.json').read_text())
METRICS=json.loads((ROOT/'results/model_results_0930_1/metrics.json').read_text())
ABL=json.loads((ANALYSIS/'ablations.json').read_text())
CATALOG=[]


def save(fig,name,title,caption,section):
    fig.savefig(FIG/name,dpi=300,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)
    with Image.open(FIG/name) as im:
        size=list(im.size)
    CATALOG.append({'file':name,'title':title,'caption':caption,'section':section,'size':size})


def footer(fig,text):
    fig.text(.02,.015,text,fontsize=9,color=GRAY,ha='left',va='bottom')


def style(ax,axis='y'):
    ax.grid(axis=axis,alpha=.18,zorder=0)
    ax.set_axisbelow(True)


def chart_eda():
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,key,title in zip(axes,['images','specimens'],['이미지 수','파일명 기준 시편 ID 수']):
        vals=[EDA['summary'][s][key] if key=='images' else len(EDA['summary'][s][key]) for s in SPLITS]
        bars=ax.bar(SPLITS,vals,color=SCOLORS,width=.58)
        ax.bar_label(bars,padding=5,fontsize=13)
        ax.set_ylim(0,max(vals)*1.2);ax.set_title(title);ax.set_ylabel('개수');style(ax)
    fig.suptitle('데이터 구성: train 70 / valid 20 / test 10',fontsize=17,y=1.01)
    fig.tight_layout(rect=[0,.06,1,.95]);footer(fig,'시편 ID는 파일명의 A번호를 기준으로 집계. test 정답 마스크는 사용하지 않음.')
    save(fig,'01_dataset_overview.png','데이터 구성','제공된 분할의 이미지 수와 파일명에서 추출한 시편 ID 수. 시편 ID 수는 분할 간 독립성을 의미하지 않는다.','1-3 EDA')

    sizes=sorted({(r['width'],r['height']) for r in EDA['records']},key=lambda x:x[0]*x[1])
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(sizes))
    for i,s in enumerate(SPLITS):
        vals=[sum(r['split']==s and (r['width'],r['height'])==d for r in EDA['records']) for d in sizes]
        bars=ax.bar(x+(i-1)*.24,vals,.23,label=s,color=SCOLORS[i]);ax.bar_label(bars,padding=3,fontsize=9)
    ax.set_xticks(x,[f'{w} × {h}' for w,h in sizes]);ax.set_xlabel('원본 해상도 (너비 × 높이, pixel)');ax.set_ylabel('이미지 수')
    ax.set_title('분할별 원본 이미지 해상도');ax.legend(frameon=False);ax.set_ylim(0,ax.get_ylim()[1]*1.14);style(ax)
    fig.tight_layout(rect=[0,.07,1,1]);footer(fig,'해상도 정보만으로 물리적 배율이나 pixel 크기가 동일하다고 가정하지 않음.')
    save(fig,'02_image_resolution.png','해상도 분포','원본 이미지의 해상도별 개수. 물리적인 배율·길이 교정값은 별도 자료가 필요하다.','1-3 EDA')

    fig,ax=plt.subplots(figsize=(10,5));y=np.arange(4)
    for i,s in enumerate(['train','valid']):
        v=100*np.array(EDA['summary'][s]['pixel_fraction'])
        bars=ax.barh(y+(i-.5)*.32,v,.3,color=SCOLORS[i],label=s)
        ax.bar_label(bars,labels=[f'{a:.2f}%' for a in v],padding=5)
    ax.set_yticks(y,NAMES);ax.invert_yaxis();ax.set_xlim(0,75);ax.set_xlabel('해당 split 전체 마스크 픽셀 중 비율 (%)')
    ax.set_title('클래스 불균형: 전체 픽셀 구성비');ax.legend(frameon=False);style(ax,'x')
    fig.tight_layout(rect=[0,.075,1,1]);footer(fig,'데이터 구성비이며 성능 지표가 아님. test 클래스 구성비는 정답이 없어 계산하지 않음.')
    save(fig,'03_class_pixel_share.png','클래스별 픽셀 구성비','전체 마스크 픽셀을 기준으로 한 클래스 비율. 큰 이미지의 픽셀 수가 더 크게 반영되며 image mIoU와는 별개의 통계이다.','1-3 EDA')

    fig,ax=plt.subplots(figsize=(11,5));rng=np.random.default_rng(349)
    for i,s in enumerate(['train','valid']):
        values=np.array([r['class_fraction'] for r in EDA['records'] if r['split']==s])*100
        positions=np.arange(4)+(i-.5)*.32
        bp=ax.boxplot([values[:,c] for c in range(4)],positions=positions,widths=.25,patch_artist=True,showfliers=False,
                      medianprops={'color':INK,'linewidth':1.6})
        for patch in bp['boxes']:patch.set_facecolor(SCOLORS[i]);patch.set_alpha(.3)
        for c in range(4):ax.scatter(positions[c]+rng.normal(0,.028,len(values)),values[:,c],s=12,color=SCOLORS[i],alpha=.45,label=s if c==0 else None,zorder=3)
    ax.set_xticks(range(4),NAMES);ax.set_xlim(-.55,3.55);ax.set_ylim(-2,102);ax.set_ylabel('이미지 내 클래스 면적 비율 (%)')
    ax.set_title('이미지별 클래스 면적 비율의 분포');ax.legend(frameon=False);style(ax)
    fig.tight_layout(rect=[0,.07,1,1]);footer(fig,'각 점은 이미지 1장. 상자: Q1–Q3, 중앙선: 중앙값, 수염: 1.5×IQR 범위.')
    save(fig,'04_class_image_area.png','이미지별 면적 비율','이미지마다 동일 가중치로 본 클래스 면적 분포. Primary Si가 없는 train 1장, valid 1장을 포함한다.','1-3 EDA')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for s,color in zip(SPLITS,SCOLORS):
        rows=[r for r in EDA['records'] if r['split']==s]
        axes[0].plot(range(256),100*np.array(EDA['mean_image_histograms'][s]),color=color,label=s,lw=1.5)
        axes[1].scatter([r['gray_mean'] for r in rows],[r['gray_std'] for r in rows],s=35,color=color,label=s,alpha=.75)
    axes[0].set(xlabel='grayscale 밝기 (0–255)',ylabel='밝기값별 평균 픽셀 비율 (%)',title='이미지별 정규화 histogram의 평균')
    axes[1].set(xlabel='이미지 평균 밝기',ylabel='이미지 밝기 표준편차',title='밝기와 대비의 이미지 간 차이')
    for ax in axes:ax.legend(frameon=False);style(ax)
    fig.tight_layout(rect=[0,.08,1,1]);footer(fig,'grayscale 변환 후 계산. histogram은 이미지별 동일 가중치; test는 입력 영상만 분석.')
    save(fig,'05_brightness_distribution.png','밝기 및 대비 분포','각 이미지의 밝기 histogram과 평균·표준편차. 노출 차이를 관찰하는 EDA이며 정규화의 성능 개선을 직접 입증하는 실험은 아니다.','1-3 EDA')

    mat=np.array([[len(set(EDA['summary'][a]['specimens'])&set(EDA['summary'][b]['specimens'])) for b in SPLITS] for a in SPLITS])
    fig,ax=plt.subplots(figsize=(7,5.8));im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=mat.max())
    for i in range(3):
        for j in range(3):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=22,color='white' if mat[i,j]>mat.max()*.5 else INK)
    ax.set_xticks(range(3),SPLITS);ax.set_yticks(range(3),SPLITS);ax.set_title('분할 사이에 공유되는 시편 ID 수');fig.colorbar(im,ax=ax,shrink=.7,label='공유 ID 수')
    fig.tight_layout(rect=[0,.10,1,1]);footer(fig,'대각선: 각 split의 고유 ID 수. 비대각선: 교집합. 파일명 A번호 기준.')
    save(fig,'06_specimen_overlap.png','시편 ID 중복','제공된 데이터 분할의 시편 ID 교집합. B4 학습 내부의 그룹 4-fold와 최종 train/valid 간 분리는 구분해야 한다.','1-3 EDA / 평가 조건')


def preprocessing():
    # Actual Model326 full-resized-image input function, with no augmentation.
    sys.path.insert(0,str(SOURCE/'experiments/model326_hybrid_fullstats/KoMaP'))
    from common import make_input,resize_gray
    train=sorted([r for r in EDA['records'] if r['split']=='train'],key=lambda r:r['gray_mean'])
    selected=[train[0],train[-1]]
    fig,axes=plt.subplots(2,4,figsize=(14.5,7.8))
    title=['Original grayscale','Raw channel','Z-score channel','CDF channel']
    limits=[(0,255),(-2,2),(-4/3,4/3),(-1.732051,1.732051)]
    for i,row in enumerate(selected):
        g=np.asarray(Image.open(row['path']).convert('L'))
        resized=resize_gray(g,3.)
        features=make_input(resized,'image_global','zcdf3')
        for j,array in enumerate([g,*features]):
            ax=axes[i,j];im=ax.imshow(array,cmap='gray',vmin=limits[j][0],vmax=limits[j][1]);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(title[j],fontsize=12);fig.colorbar(im,ax=ax,shrink=.65,pad=.02)
        axes[i,0].set_ylabel(f"{row['image']}\n원본 평균 {row['gray_mean']:.1f}",fontsize=10)
    fig.suptitle('Model326 입력 표현: 밝기가 낮은 영상과 높은 영상',fontsize=17,y=.99)
    fig.tight_layout(rect=[0,.075,1,.94]);footer(fig,'train 내 평균 밝기 최솟값·최댓값 영상을 선택. 3배 확대 후 실제 입력 함수를 적용; 채널별 공통 표시 범위 사용.')
    save(fig,'07_raw_zscore_cdf.png','Raw/Z-score/CDF 입력 예시','학습 영상 중 평균 밝기가 가장 낮고 높은 두 장을 사전에 정한 기준으로 선택했다. Model326 코드로 3배 확대 후 표현을 계산했으며, 표시용 grayscale 범위는 각 채널에서 두 영상에 동일하게 적용했다. 증강을 적용하지 않은 입력 예시이다.','2-1 전처리')
    (ANALYSIS/'preprocessing_examples.json').write_text(json.dumps([{'image':r['image'],'mean':r['gray_mean']} for r in selected],ensure_ascii=False,indent=2))


def diagram(title, size=(14,8)):
    fig,ax=plt.subplots(figsize=size);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.suptitle(title,fontsize=19,color=INK,y=.975)
    return fig,ax


def box(ax,x,y,w,h,text,color='#EDF3FC',edge=BLUE,fontsize=12):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.012',
                    linewidth=1.3,edgecolor=edge,facecolor=color,zorder=2)
    ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color=INK,linespacing=1.55,zorder=3)


def arrow(ax,a,b,style='-|>',color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=15,linewidth=1.4,color=color,zorder=1))


def diagrams():
    fig,ax=diagram('전처리와 추론 절차: B4 / HRNet 분기')
    box(ax,.34,.82,.32,.10,'입력 영상과 제공된 데이터 분할',fontsize=13)
    for cx in [.24,.76]:arrow(ax,(.5,.82),(cx,.74))
    box(ax,.03,.60,.42,.14,'B4: Model322 / 326 / 329\n3배 확대 또는 너비 4000 조정')
    box(ax,.55,.60,.42,.14,'HRNet: Model336 / 347\nRGB + ImageNet 정규화')
    arrow(ax,(.24,.60),(.24,.53));arrow(ax,(.76,.60),(.76,.53))
    box(ax,.03,.37,.42,.16,'학습: 1536 패치 + 증강\n326/329: 전체 영상 통계로 Raw/Z/CDF\n추론: 전체 영상 forward + D4(8개)',fontsize=11)
    box(ax,.55,.37,.42,.16,'학습: 512 패치, input scale 2\n추론: native 256 타일 → 512 입력\noverlap 170 + flip TTA(4개)',fontsize=11)
    arrow(ax,(.24,.37),(.24,.30));arrow(ax,(.76,.37),(.76,.30))
    box(ax,.03,.18,.42,.12,'원본 해상도로 확률 복원\n4개 fold의 확률 평균')
    box(ax,.55,.18,.42,.12,'원본 해상도로 확률 복원\nGaussian window로 타일 결합')
    arrow(ax,(.24,.18),(.5,.10));arrow(ax,(.76,.18),(.5,.10))
    ax.text(.5,.055,'클래스 채널 순서 통일 → Model349 확률 결합',ha='center',fontsize=13,color=BLUE)
    footer(fig,'학습 증강과 test-time augmentation(TTA)은 별도 처리. 마스크 크기 변경에는 nearest-neighbor를 사용.')
    save(fig,'08_preprocessing_flow.png','전처리·추론 흐름','B4와 HRNet 분기의 전처리 및 추론 차이. ResNet 기반 기본 분기는 별도 구조도와 설정표를 참조한다.','2-2 전처리 절차')

    fig,ax=diagram('Model349 전체 구조: 클래스별 확률 앙상블',(15,9))
    box(ax,.36,.84,.28,.09,'현미경 입력 이미지',fontsize=14)
    centers=[.125,.375,.625,.875]
    texts=['기본 ResNet34 계열\ncompact U-Net++ / ASPP\n고해상도·스타일·TENT',
           'Model269\nResNet34 + ASPP\n원본 해상도 native stem',
           'Model322 / 326 / 329\nEfficientNet-B4 + U-Net\n각 4-fold 확률 평균',
           'Model336 / 347\nHRNet-W48 + U-Net++\n경계 보조 학습']
    for c,t in zip(centers,texts):
        arrow(ax,(.5,.84),(c,.75));box(ax,c-.11,.57,.22,.18,t,fontsize=11)
        arrow(ax,(c,.57),(c,.49))
    ax.plot([.125,.875],[.49,.49],color=GRAY,lw=1.3);arrow(ax,(.5,.49),(.5,.43))
    box(ax,.13,.29,.74,.14,'Model339 → Model341 → Model346 → Model349\n클래스 확률 교체 → 매번 4개 클래스 전체 재정규화',color='#EAF7F5',edge=TEAL,fontsize=13)
    arrow(ax,(.5,.29),(.5,.23))
    box(ax,.13,.12,.74,.11,'밝기 prior + 클래스 bias → Gaussian / bilateral 필터\n→ argmax → Primary Si 작은 성분 및 1% 면적 규칙',fontsize=12)
    arrow(ax,(.5,.12),(.5,.06));ax.text(.5,.03,'최종 4클래스 segmentation mask',ha='center',fontsize=13,color=BLUE)
    footer(fig,'고정된 로컬 실행 설정. 최종 결합은 학습된 gating network가 아닌 클래스별 계수와 규칙을 사용.')
    save(fig,'09_model349_pipeline.png','Model349 전체 구조','원본 체크포인트를 사용하는 여러 분기의 확률을 순차 결합한 후 영상 통계와 공간 보정을 적용한다. Model349 성능은 이 전체 파이프라인의 결과이다.','3-2 모델링 방법')

    fig,ax=diagram('ResNet34 계열 세부 구조와 Model269의 native stem',(15,9))
    box(ax,.02,.74,.18,.13,'입력\n3 × H × W')
    box(ax,.26,.70,.32,.21,'ResNet34 encoder\n채널: 64 / 64 / 128 / 256 / 512\n해상도: 1/2 → 1/32',fontsize=12)
    arrow(ax,(.20,.80),(.26,.80))
    box(ax,.65,.70,.32,.21,'Residual ASPP\n512 → 5개 분기 × 128\nconcat 640 → projection 512',fontsize=12)
    arrow(ax,(.58,.80),(.65,.80))
    box(ax,.65,.40,.32,.19,'compact nested U-Net++\n중첩 skip connection\n입력 해상도로 upsampling',fontsize=12)
    arrow(ax,(.81,.70),(.81,.59));arrow(ax,(.43,.70),(.65,.51))
    box(ax,.27,.40,.31,.19,'입력 크기의 native stem (269)\nConv 3→32→32, stride 1\nzero-init Conv 32→32',color='#EAF7F5',edge=TEAL,fontsize=11)
    arrow(ax,(.11,.74),(.11,.50));arrow(ax,(.11,.50),(.27,.50))
    box(ax,.40,.12,.46,.16,'Head 내부에서 native 특징 더하기 (269)\n32채널 특징 → 1×1 Conv → 4 logits',fontsize=12)
    arrow(ax,(.43,.40),(.53,.28));arrow(ax,(.81,.40),(.73,.28))
    ax.text(.03,.27,'ASPP 분기\nRGB 계열: 1×1 + d=6,12,18 + pooling\n269 계열: 1×1 + d=2,4,8 + pooling',fontsize=10,color=GRAY,va='top',linespacing=1.8)
    footer(fig,'ASPP 출력: x + s·Project(concat), s는 0 초기화. native stem은 Model269 계열에 추가된 경로.')
    save(fig,'10_resnet_aspp_native.png','ResNet34·ASPP·native stem','기본 ResNet 계열의 encoder/decoder와 Model269의 추가 native stem을 함께 표시했다. RGB 계열과 표현 변환 계열의 dilation 설정은 다르다.','3-1 알고리즘 / 3-2 모델링')

    fig,ax=diagram('B4 및 HRNet 전문가 구조',(14,8))
    box(ax,.03,.76,.41,.14,'Model322 / 326 / 329\n3채널 표현 → EfficientNet-B4 encoder',fontsize=12)
    box(ax,.56,.76,.41,.14,'Model336 / 347\n3채널 영상 → HRNet-W48 encoder',fontsize=12)
    box(ax,.03,.49,.41,.15,'U-Net decoder\nupsampling + encoder skip 특징 결합',fontsize=12)
    box(ax,.56,.49,.41,.15,'U-Net++ decoder\n다중 해상도 특징 + 중첩 skip 결합',fontsize=12)
    arrow(ax,(.235,.76),(.235,.64));arrow(ax,(.765,.76),(.765,.64))
    box(ax,.03,.20,.41,.16,'Segmentation head → 4 logits\nSoftmax → TTA·fold 평균\nB4 총 12개 checkpoint',fontsize=12)
    arrow(ax,(.235,.49),(.235,.36))
    box(ax,.55,.20,.22,.16,'Segmentation head\n4 logits → Softmax\n추론에 사용',fontsize=11)
    box(ax,.80,.20,.18,.16,'Boundary head\n3×3 Conv → 1채널\n학습에 사용',color='#FFF4E8',edge=ORANGE,fontsize=10)
    arrow(ax,(.765,.49),(.66,.36));arrow(ax,(.765,.49),(.89,.36))
    ax.text(.765,.105,'Model347: Model336 last.pt에서\n클래스 가중 CE로 fine-tuning',ha='center',fontsize=11,color=GRAY)
    footer(fig,'B4는 SMP U-Net, HRNet은 SMP U-Net++. 기존 ResNet의 compact U-Net++와 구현을 구분.')
    save(fig,'11_b4_hrnet_architecture.png','B4·HRNet 전문가 구조','B4 계열과 HRNet 계열의 encoder, decoder, 출력 head 차이. Boundary head는 학습 보조용이다.','3-1 알고리즘 / 3-2 모델링')

    fig,ax=plt.subplots(figsize=(11,6))
    g=np.array([[.7,1.,.6,.7],[.5,1.,0,0],[0,.25,.75,.75],[0,0,.25,0]])
    sources=[['326','326','322','269'],['329','329','',''],['','336','336','336'],['','','347','']]
    im=ax.imshow(g,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for r in range(4):
        for c in range(4):
            ax.text(c,r,f'Model{sources[r][c]}\ngamma = {g[r,c]:.2f}' if g[r,c] else '추가 없음',ha='center',va='center',fontsize=12,color='white' if g[r,c]>.6 else INK)
    ax.set_xticks(range(4),NAMES);ax.set_yticks(range(4),['Model339','Model341','Model346','Model349']);ax.set_title('순차 클래스별 확률 결합 계수',pad=15)
    fig.colorbar(im,ax=ax,shrink=.8,label='해당 단계의 gamma')
    fig.tight_layout(rect=[0,.12,1,1]);footer(fig,'Model339 내부 순서: Primary Si → Al3Ni → Eutectic Si → Al. 이후 클래스 인덱스 순 처리.\n각 교체 직후 재정규화하므로 계수의 단순 곱은 최종 모델 가중치가 아님.')
    save(fig,'12_classwise_fusion.png','클래스별 결합 계수','선택된 전문가와 각 단계의 γ. 표에 보이는 γ는 해당 단계에서 사용한 혼합 계수이며 전체 시스템의 선형 가중치가 아니다.','3-2 모델링 방법')

    fig,ax=diagram('평가 지표: 이미지별 4클래스 IoU를 먼저 평균',(12,6))
    for c,(label,value) in enumerate(zip(NAMES,['0.80','0.00','0.80','0.80'])):
        box(ax,.03+c*.25,.63,.19,.21,f'{label}\nIoU = {value}',fontsize=12)
        arrow(ax,(.125+c*.25,.63),(.5,.48))
    box(ax,.21,.31,.58,.17,'이미지 1장의 mIoU = (0.8 + 0 + 0.8 + 0.8) / 4\n= 60%',color='#EAF7F5',edge=TEAL,fontsize=13)
    arrow(ax,(.5,.31),(.5,.20));ax.text(.5,.125,'validation 20장의 image mIoU를 동일 가중치로 평균',ha='center',fontsize=14)
    footer(fig,'설명용 예시. 정답·예측 모두에 없는 클래스(union=0)의 IoU는 0. 픽셀 정확도와 다른 지표.')
    save(fig,'18_image_miou_definition.png','평가 지표 설명','실제 측정값이 아닌 설명용 예시이다. 빈 클래스의 IoU를 0으로 두고 네 클래스 평균, 이후 20장 평균을 구한다.','3-3 성능')


def performance():
    values=[]
    for dirname in ['model339_joint_gamma_route_search','model341_model329_joint_route_search','model346_model336_into_341_route','model349_model347_into_346_route']:
        d=json.loads((RUN/'outputs'/dirname/'results.json').read_text())
        values.append(100*(d['best_image_empty_zero']['image_mean_miou'] if 'best_image_empty_zero' in d else d['raw_best_valid_image_mean_miou_empty0']))
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),gridspec_kw={'width_ratios':[1.1,1]})
    bars=axes[0].bar(['339','341','346','349'],values,color=[GRAY,BLUE,TEAL,ORANGE],width=.58)
    axes[0].bar_label(bars,labels=[f'{v:.4f}%' for v in values],padding=5);axes[0].set(ylim=(0,100),ylabel='Validation image mIoU (%)',xlabel='Model',title='동일 실행의 단계별 점수');style(axes[0])
    delta=np.diff(values);y=np.arange(3)
    bars=axes[1].barh(y,delta,color=[BLUE,TEAL,ORANGE],height=.55)
    axes[1].bar_label(bars,labels=[f'+{v:.6f}%p' for v in delta],padding=5,fontsize=10)
    axes[1].set_yticks(y,['339 → 341','341 → 346','346 → 349']);axes[1].invert_yaxis();axes[1].set(xlim=(0,max(delta)*1.6),xlabel='image mIoU 변화 (%p)',title='직전 단계 대비 변화');style(axes[1],'x')
    fig.tight_layout(rect=[0,.09,1,1]);footer(fig,'단계별 선택 기록. 개별 요소의 독립적인 기여를 분리한 재학습 ablation은 아님.')
    save(fig,'13_stage_performance.png','단계별 성능','Model339·341·346·349의 동일 실행 기록. 점수는 image mIoU(empty=0), 차이는 percentage point이다.','3-3 성능')

    vals=np.array([[r['class_iou'][c] for c in NAMES] for r in METRICS['images']])*100
    fig,ax=plt.subplots(figsize=(10,5));means=vals.mean(0)
    bars=ax.bar(NAMES,means,color=PCOLORS,width=.62);ax.bar_label(bars,labels=[f'{v:.4f}%' for v in means],padding=6)
    ax.axhline(METRICS['image_miou']*100,color=INK,ls='--',lw=1.4,label=f"전체 image mIoU {METRICS['image_miou']*100:.4f}%")
    ax.set(ylim=(0,105),ylabel='이미지별 클래스 IoU의 평균 (%)',title='Model349 클래스별 성능');ax.legend(frameon=False,loc='lower right');style(ax)
    fig.tight_layout(rect=[0,.08,1,1]);footer(fig,'각 클래스의 이미지별 IoU를 20장 평균. 빈 클래스 IoU=0; 전체 픽셀을 합쳐 계산한 IoU가 아님.')
    save(fig,'14_class_iou.png','클래스별 IoU','Al3Ni, Primary Si, Al, Eutectic Si의 이미지별 IoU 평균. 점선은 네 클래스 평균과 같은 전체 image mIoU이다.','3-3 성능')

    rows=sorted(METRICS['images'],key=lambda r:r['image_miou'])
    fig,ax=plt.subplots(figsize=(11,9));v=[r['image_miou']*100 for r in rows]
    bars=ax.barh(range(20),v,color=[ORANGE if i in [0,19] else BLUE for i in range(20)],height=.65)
    ax.bar_label(bars,labels=[f'{x:.2f}' for x in v],padding=4,fontsize=10)
    ax.set_yticks(range(20),[r['image'] for r in rows]);ax.set(xlim=(0,104),xlabel='image mIoU (%)',title='Validation 20장: 이미지별 성능')
    ax.axvline(METRICS['image_miou']*100,color=TEAL,ls='--',label=f"평균 {METRICS['image_miou']*100:.4f}%");ax.legend(frameon=False,loc='lower right');style(ax,'x')
    fig.tight_layout(rect=[0,.04,1,1]);footer(fig,'점수 오름차순 정렬. 모든 이미지의 4개 클래스에 동일 가중치 적용.')
    save(fig,'15_per_image_miou.png','이미지별 mIoU','검증 이미지 20장 전체의 점수와 평균선. 가장 낮은 이미지와 가장 높은 이미지를 강조했다.','3-3 성능')

    mat=np.array([[r['class_iou'][c]*100 for c in NAMES] for r in rows])
    fig,ax=plt.subplots(figsize=(10.5,10));im=ax.imshow(mat,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    for r in range(20):
        for c in range(4):ax.text(c,r,f'{mat[r,c]:.1f}',ha='center',va='center',fontsize=10,color='white' if mat[r,c]>=60 else INK)
    ax.set_xticks(range(4),NAMES);ax.set_yticks(range(20),[r['image'] for r in rows]);ax.set_title('이미지 × 클래스 IoU (%)',pad=12);fig.colorbar(im,ax=ax,shrink=.75,label='IoU (%)')
    fig.tight_layout(rect=[0,.06,1,1]);footer(fig,'이미지별 평균 점수 오름차순. Primary Si=0인 항목도 제외하지 않고 평균에 포함.')
    save(fig,'16_per_image_class_iou.png','이미지별 클래스 IoU heatmap','전체 평균에서 드러나지 않는 특정 클래스·이미지의 낮은 성능을 확인할 수 있다.','3-3 성능')

    cases=ABL['cases'];fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1.45,1]})
    y=np.arange(len(cases));colors=[ORANGE]+[BLUE]*6
    bars=axes[0].barh(y,[r['image_miou']*100 for r in cases],color=colors,height=.60)
    axes[0].bar_label(bars,labels=[f"{r['image_miou']*100:.4f}%" for r in cases],padding=4,fontsize=10)
    axes[0].set_yticks(y,[r['label'] for r in cases]);axes[0].invert_yaxis();axes[0].set(xlim=(0,105),xlabel='Validation image mIoU (%)',title='동일한 저장 확률을 사용한 비교');style(axes[0],'x')
    d=[r['delta_pp'] for r in cases];bars=axes[1].barh(y,d,color=colors,height=.60)
    axes[1].axvline(0,color=INK,lw=1);axes[1].set_yticks(y,['']*len(cases));axes[1].invert_yaxis();axes[1].set(xlabel='Model349 전체 대비 변화 (%p)',title='원래 설정 대비 차이')
    lo,hi=min(d),max(d);pad=max((hi-lo)*.6,.3);axes[1].set_xlim(lo-pad,hi+pad)
    for i,v in enumerate(d):axes[1].text(v+(.04 if v>=0 else -.04),i,f'{v:+.4f}',ha='right' if v<0 else 'left',va='center',fontsize=10)
    style(axes[1],'x');fig.tight_layout(rect=[0,.12,1,1])
    footer(fig,'재학습·계수 재탐색 없이 고정된 확률과 설정으로 산출한 사후 분석.\n균등 평균: 기본 확률 + 269/322/326/329/336/347의 7개 확률원을 각 1/7로 결합; 모든 경우 empty=0.')
    save(fig,'17_frozen_cache_ablation.png','저장 확률 기반 후처리·결합 비교','기존 Model349와 같은 확률을 사용하고 명시된 요소만 바꿨다. 20장 전체에서 원래 설정이 기존 마스크와 픽셀 단위로 일치함을 확인했다. 새 모델 학습이나 검증셋에서의 추가 파라미터 탐색은 하지 않았다.','3-3 성능 / 추가 분석')


if __name__=='__main__':
    PACKAGE.mkdir(parents=True,exist_ok=False)
    FIG.mkdir()
    for fn in tqdm([chart_eda,preprocessing,diagrams,performance],desc='Create report figures',unit='group'):
        fn()
    assert len(CATALOG)==18,len(CATALOG)
    (ANALYSIS/'figure_catalog.json').write_text(json.dumps(CATALOG,ensure_ascii=False,indent=2))
    print('FIGURES_DONE',len(CATALOG),flush=True)
