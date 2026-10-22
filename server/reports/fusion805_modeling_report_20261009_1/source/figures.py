"""Evidence-derived publication figures; editable SVG plus 300-dpi PNG."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Patch
from tqdm import tqdm
from analyze import PACKAGE,ROOT,RESULT,EXP,RUN,CLASSES,PALETTE,read,write,decode

font_manager.fontManager.addfont(PACKAGE/'source/NanumGothic-Regular.ttf')
plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':10,
 'axes.titlesize':14,'axes.labelsize':10,'figure.facecolor':'white','savefig.facecolor':'white',
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
BLUE='#2455A4';TEAL='#168D91';ORANGE='#D88437';INK='#20324A';GRAY='#68788A';RED='#B94848'
COLORS={'R':GRAY,'E':'#7A60A0','stage1':TEAL,'stage2':BLUE}
EDA=read(PACKAGE/'evidence/eda.json');M=read(PACKAGE/'evidence/audited_metrics.json')
CH=read(PACKAGE/'evidence/pixel_changes.json');TIME=read(PACKAGE/'evidence/runtime.json')
CAT=[]

def save(fig,name,title,caption):
    fig.savefig(PACKAGE/'figures'/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=.14)
    fig.savefig(PACKAGE/'figures'/f'{name}.svg',bbox_inches='tight',pad_inches=.14)
    plt.close(fig);CAT.append(dict(id=name,title=title,caption=caption,file=f'figures/{name}.png',svg=f'figures/{name}.svg'))
def footer(fig,text):fig.text(.02,.012,text,fontsize=8,color=GRAY,va='bottom')
def grid(ax,axis='y'):ax.grid(axis=axis,alpha=.18);ax.set_axisbelow(True)
def box(ax,x,y,w,h,text,color=BLUE,size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',fc=color+'15',ec=color,lw=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK,linespacing=1.5)
def arrow(ax,a,b,color=GRAY):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.3,color=color))

def eda_figures():
    fig,axs=plt.subplots(1,2,figsize=(10,4));splits=['train','valid','test'];cols=[BLUE,TEAL,ORANGE]
    for ax,key,title in zip(axs,['images','specimens'],['이미지 수','파일명 기준 시편 ID 수']):
        vals=[EDA['summary'][s][key] if key=='images' else len(EDA['summary'][s][key]) for s in splits]
        bars=ax.bar(splits,vals,color=cols,width=.6);ax.bar_label(bars,padding=4);ax.set_ylim(0,max(vals)*1.18);ax.set_title(title);grid(ax)
    fig.tight_layout(rect=[0,.07,1,1]);footer(fig,'test 10장은 입력 영상만 분석. 정답 마스크는 제공되지 않음.')
    save(fig,'01_dataset','데이터 구성','실제 입력 파일을 집계했다. train70, valid20, test10이며 시편 수는 A번호 기준이다. [S02]')
    dims=sorted({(r['width'],r['height']) for r in EDA['records']},key=lambda t:t[0]*t[1])
    fig,ax=plt.subplots(figsize=(10,4.2));x=np.arange(len(dims))
    for i,s in enumerate(splits):
        vals=[sum(r['split']==s and (r['width'],r['height'])==d for r in EDA['records']) for d in dims]
        b=ax.bar(x+(i-1)*.24,vals,.23,label=s,color=cols[i]);ax.bar_label(b,padding=3,fontsize=9)
    ax.set_xticks(x,[f'{w} × {h}' for w,h in dims]);ax.set_ylabel('이미지 수');ax.set_title('원본 해상도 분포');ax.legend(frameon=False);grid(ax)
    fig.tight_layout(rect=[0,.08,1,1]);footer(fig,'너비 × 높이, pixel. 물리적 배율이나 길이 교정값으로 해석하지 않음.')
    save(fig,'02_resolution','원본 해상도','분할별 해상도 분포. 하나의 고정 크기로 제공된 데이터가 아니므로 원본 좌표 복원이 필요하다. [S02]')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for i,s in enumerate(['train','valid']):
        vals=100*np.array(EDA['summary'][s]['pixel_share']);b=axs[0].barh(np.arange(4)+(i-.5)*.3,vals,.29,color=cols[i],label=s)
        axs[0].bar_label(b,labels=[f'{v:.2f}' for v in vals],padding=3,fontsize=8)
        f=100*np.array([r['class_fraction'] for r in EDA['records'] if r['split']==s])
        for c in range(4):axs[1].scatter(np.full(len(f),c)+(i-.5)*.2,f[:,c],alpha=.5,s=12,color=cols[i])
    axs[0].set_yticks(range(4),CLASSES);axs[0].invert_yaxis();axs[0].set_xlim(0,77);axs[0].set_xlabel('전체 픽셀 구성비 (%)');axs[0].legend(frameon=False)
    axs[1].set_xticks(range(4),CLASSES,rotation=15);axs[1].set_ylabel('이미지 내 클래스 면적 (%)');axs[1].set_ylim(-3,103)
    axs[0].set_title('전체 마스크 픽셀 기준');axs[1].set_title('이미지별 면적 비율 분포')
    for ax in axs:grid(ax)
    fig.tight_layout(rect=[0,.08,1,1]);footer(fig,'데이터 구성 통계이며 평가 점수가 아님. image mIoU는 이미지·클래스에 동일 가중치를 부여함.')
    save(fig,'03_classes','클래스 불균형','좌측은 전체 픽셀 기준, 우측은 각 이미지의 클래스 면적 비율이다. 두 통계의 가중 방식이 다르다. [S02]')
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    for s,c in zip(splits,cols):
        rows=[r for r in EDA['records'] if r['split']==s]
        axs[0].plot(100*np.array(EDA['histograms'][s]),label=s,color=c)
        axs[1].scatter([r['mean'] for r in rows],[r['std'] for r in rows],label=s,color=c,s=26,alpha=.7)
    axs[0].set(xlabel='Grayscale (0–255)',ylabel='평균 픽셀 비율 (%)',title='이미지별 histogram 평균')
    axs[1].set(xlabel='영상 평균 밝기',ylabel='영상 밝기 표준편차',title='밝기·대비 분포')
    for ax in axs:ax.legend(frameon=False);grid(ax)
    fig.tight_layout(rect=[0,.08,1,1]);footer(fig,'PIL grayscale, 이미지별 동일 가중치. 정규화의 성능 기여를 직접 입증하는 실험은 아님.')
    save(fig,'04_brightness','밝기 및 대비','영상 사이의 밝기·대비 차이를 확인했다. 정규화와 서로 다른 입력 표현을 사용하는 설계의 관찰 근거다. [S02]')
    mat=np.array([[len(set(EDA['summary'][a]['specimens'])&set(EDA['summary'][b]['specimens'])) for b in splits] for a in splits])
    fig,ax=plt.subplots(figsize=(6.6,4.7));ax.imshow(mat,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=23,color='white' if mat[i,j]>30 else INK)
    ax.set_xticks(range(3),splits);ax.set_yticks(range(3),splits);ax.set_title('분할 사이의 공통 시편 ID 수')
    fig.tight_layout(rect=[0,.11,1,1]);footer(fig,'대각선: 고유 시편 수. 내부 OOF 그룹 분리와 공식 train/valid의 분리는 별개.')
    save(fig,'05_overlap','시편 ID 중복','train/valid는 파일명 기준 7개 시편 ID를 공유한다. 현재 valid20을 완전한 외부 시편 평가로 해석하지 않는다. [S02]')

def methods():
    p=EXP/'data/valid/images/A63_step1_sample4_image.png';g=np.array(Image.open(p).convert('L'),np.float32)/255
    z=(g-g.mean())/(g.std()+1e-6);bins=np.clip((g*255).astype(int),0,255);cdf=np.cumsum(np.bincount(bins.ravel(),minlength=256));cdf=cdf/cdf[-1]
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax,a,t in zip(axs,[g,np.clip(z,-4,4),cdf[bins]],['Raw grayscale','Z-score (표시 범위 ±4)','밝기 누적분포 CDF']):ax.imshow(a,cmap='gray');ax.set_title(t);ax.axis('off')
    fig.tight_layout(rect=[0,.1,1,1]);footer(fig,'A63_step1_sample4. 그림은 표현의 시각적 비교이며 실제 네트워크 채널 scaling은 본문 식을 따름.')
    save(fig,'06_representation','Raw/Z-score/CDF 표현','같은 원본에서 절대 밝기·상대 밝기·밝기 순위를 구성한다. CDF의 표시 예시와 실제 구현의 세부 변환은 구분한다. [S02, S05]')
    fig,ax=plt.subplots(figsize=(11,6.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.32,.87,.36,.09,'원본 현미경 영상',INK)
    box(ax,.03,.61,.42,.16,'R: exh14, 14 checkpoints\nResNet34 계열 + B4 + HRNet\n기존 고정 route / 후처리',BLUE)
    box(ax,.55,.61,.42,.16,'E: 12 checkpoints\nB4 4-fold + HRNet 2 seeds × 4-fold\n0.50 + 0.25 + 0.25',TEAL)
    arrow(ax,(.42,.87),(.24,.78));arrow(ax,(.58,.87),(.76,.78))
    box(ax,.12,.34,.42,.15,'Stage 1: R + E (대표)\nE가 Al3Ni 또는 Al → E label\n그 외 → R label',TEAL)
    arrow(ax,(.24,.61),(.28,.49));arrow(ax,(.69,.61),(.49,.49))
    box(ax,.64,.34,.32,.15,'F: 잔차 보정기 1개\nE log-prob + gray\ntrain70 OOF로 학습',ORANGE)
    arrow(ax,(.8,.61),(.8,.49))
    box(ax,.28,.07,.48,.15,'Stage 2: R + E + F (비교)\nF가 Al → Al, 그 외 → Stage 1',BLUE)
    arrow(ax,(.34,.34),(.43,.22));arrow(ax,(.79,.34),(.64,.22))
    ax.set_title('Fusion805 전체 추론 구조: 총 27개 가중치',pad=10)
    save(fig,'07_pipeline','Fusion805 전체 구조','R은 14개, E는 12개, F는 1개 checkpoint다. stage1은 F를 사용하지 않으며 stage2의 최종 규칙도 모든 F label을 덮어쓰는 방식이 아니다. [S03–S06]')
    fig,axs=plt.subplots(1,3,figsize=(12,5.0))
    recipes=[('R: ResNet34 분기',['입력 / 표현 변환','ResNet34 encoder','중첩 decoder + ASPP','4-class logits'],BLUE),('E: B4 / HRNet',['Raw·Z·CDF / RGB','B4 / HRNet-W48','U-Net / U-Net++','4-class probabilities'],TEAL),('F: 경량 잔차',['4 logits + 1 gray','Conv 5→32, SiLU','Conv 32→32, SiLU','Conv 32→4, tanh'],ORANGE)]
    for ax,(title,nodes,color) in zip(axs,recipes):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,fontsize=12)
        for i,t in enumerate(nodes):
            y=.77-i*.21;box(ax,.06,y,.88,.13,t,color,10)
            if i<3:arrow(ax,(.5,y),(.5,y-.075))
    fig.tight_layout(rect=[0,.09,1,1]);footer(fig,'도식은 특징 흐름 요약. R의 native stem, HRNet의 학습용 boundary head 등 세부 구조는 본문·소스 참조.')
    save(fig,'08_architecture','신경망 계열 비교','세 계열의 계산 흐름을 요약했다. F는 11,876 parameters이며 최종 convolution을 0으로 초기화한다. [S04–S06]')
    fig,ax=plt.subplots(figsize=(11,5.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.03,.76,.2,.15,'train70\n59 specimen IDs',BLUE)
    box(ax,.36,.76,.27,.15,'시편 단위 4-fold\n52/18, 52/18, 53/17, 53/17',TEAL,10)
    box(ax,.76,.76,.2,.15,'valid20\n최종 평가용',GRAY)
    arrow(ax,(.23,.84),(.36,.84))
    box(ax,.03,.42,.27,.18,'각 fold의 학습 부분\nB4 + HR seed1 + HR seed2\n해당 시편은 제외',BLUE,10)
    box(ax,.4,.42,.25,.18,'해당 fold held-out 예측\n0.50 / 0.25 / 0.25\nOOF 70장을 연결',TEAL,10)
    box(ax,.75,.42,.21,.18,'4-fold 평균 E\nvalid20 추론\n최종 점수 계산',GRAY,10)
    arrow(ax,(.43,.76),(.19,.60));arrow(ax,(.3,.51),(.4,.51));arrow(ax,(.86,.76),(.86,.60))
    box(ax,.3,.07,.43,.17,'F 학습: OOF log-prob + gray + train GT\nseed 81349 / 8 epochs / 1,120 updates',ORANGE,10)
    arrow(ax,(.52,.42),(.52,.24));ax.set_title('OOF를 이용한 보정기 학습과 평가 데이터의 역할')
    save(fig,'09_oof','OOF 학습 절차','F의 각 학습 입력은 해당 이미지를 학습에서 제외한 전문가의 예측이다. valid20 정답은 F 학습에 사용하지 않는다. 공식 train/valid 시편 중복과는 별개의 통제다. [S03, S06]')
    fig,ax=plt.subplots(figsize=(10,3.8));v=[80,0,80,80];b=ax.bar(CLASSES,v,color=np.array(PALETTE)/255,width=.55);ax.bar_label(b,labels=['80%','0% (부재 예시)','80%','80%'],padding=6)
    ax.set_ylim(0,105);ax.set_ylabel('클래스 IoU (%)');ax.set_title('지표 설명용 예시: (0.8 + 0 + 0.8 + 0.8) / 4 = 60%');grid(ax)
    fig.tight_layout(rect=[0,.13,1,1]);footer(fig,'실측 결과가 아닌 산술 예시. absent class는 항상 해당 클래스의 위치에 0을 넣음; 네 클래스 모두 같은 규칙.')
    save(fig,'10_metric','image mIoU의 평균 규칙','빈 union은 0으로 처리하고 네 클래스 평균 후 이미지 20장을 평균한다. 그림의 수치는 설명용이며 실제 클래스 성능이 아니다. [S01]')

def performance():
    fig,axs=plt.subplots(1,2,figsize=(11,4.2));keys=['R','E','stage1','stage2'];vals=[M[k]['image_miou']*100 for k in keys]
    b=axs[0].bar(['R','E','R+E','R+E+F'],vals,color=[COLORS[k] for k in keys]);axs[0].bar_label(b,labels=[f'{v:.4f}%' for v in vals],padding=5);axs[0].set_ylim(0,100);axs[0].set_ylabel('image mIoU (%)');axs[0].set_title('현재 재학습 실행, valid20')
    d=[(M['stage1']['image_miou']-M['R']['image_miou'])*100,(M['stage2']['image_miou']-M['stage1']['image_miou'])*100]
    b=axs[1].bar(['R → R+E','R+E → R+E+F'],d,color=[TEAL,BLUE]);axs[1].bar_label(b,labels=[f'+{v:.6f}%p' for v in d],padding=6);axs[1].set_ylim(0,max(d)*1.3);axs[1].set_ylabel('image mIoU 변화 (%p)');axs[1].set_title('같은 실행 내 단계별 변화')
    for ax in axs:grid(ax)
    fig.tight_layout(rect=[0,.09,1,1]);footer(fig,'E 단독은 저장된 E 확률에서 평가. F의 작은 개선만으로 독립 데이터의 일반화 이득을 주장하지 않음.')
    save(fig,'11_stage_scores','현재 실행의 구성별 성능','R/E/두 결합 단계의 valid20 image mIoU. 우측은 같은 부모 가중치에서 결합 단계를 추가했을 때의 변화다. [S03, S07]')
    fig,ax=plt.subplots(figsize=(10,4.5));labels=['GitHub\n고정 가중치 fresh R','GitHub\n고정 가중치 fresh R+E','로컬 재학습\nR','로컬 재학습\nR+E','로컬 재학습\nR+E+F']
    vals=[80.2309863330,80.4744236018,M['R']['image_miou']*100,M['stage1']['image_miou']*100,M['stage2']['image_miou']*100]
    bars=ax.bar(labels,vals,color=[GRAY,ORANGE,GRAY,TEAL,BLUE]);ax.bar_label(bars,labels=[f'{v:.6f}%' for v in vals],padding=5,fontsize=9);ax.set_ylim(0,100);ax.set_ylabel('image mIoU (%)');grid(ax)
    ax.set_title('GitHub 고정 가중치 재추론과 로컬 재학습의 구분');fig.tight_layout(rect=[0,.12,1,1]);footer(fig,'서로 다른 가중치·실행. 원인 통제 실험이 아니며 두 실행의 점수 차이를 특정 변경의 효과로 해석하지 않음.')
    save(fig,'12_provenance_scores','출처별 점수 비교','GitHub 최신 fresh 추론 기록과 로컬 재학습 결과를 구분했다. GitHub의 저장 확률 기반 stage2 80.500689%는 본문의 별도 표에 제시한다. [S03, S08, S09]')
    fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(4)
    for j,k in enumerate(keys):
        vals=100*np.array(M[k]['class_iou']);b=ax.bar(x+(j-1.5)*.19,vals,.18,color=COLORS[k],label=k);ax.bar_label(b,fmt='%.2f',padding=3,fontsize=7)
    ax.set_xticks(x,CLASSES);ax.set_ylim(0,105);ax.set_ylabel('이미지별 class IoU 평균 (%)');ax.legend(ncol=4,frameon=False,loc='upper left');grid(ax)
    ax.set_title('클래스별 결과: 각 이미지의 IoU를 20장 평균');fig.tight_layout()
    save(fig,'13_class_iou','클래스별 평균 IoU','전체 픽셀을 합쳐 계산한 IoU가 아니다. 빈 클래스도 IoU 0으로 포함한 이미지별 평균이다. [S07]')
    rows=sorted(M['stage1']['images'],key=lambda r:r['image_miou']);names=[r['image'] for r in rows]
    fig,ax=plt.subplots(figsize=(10,7))
    for k in ['R','stage1','stage2']:
        by={r['image']:r['image_miou']*100 for r in M[k]['images']};ax.plot([by[n] for n in names],range(20),'o-',ms=4,lw=.8,label=k,color=COLORS[k])
    ax.set_yticks(range(20),names);ax.set_xlim(45,95);ax.set_xlabel('image mIoU (%)');ax.legend(frameon=False);ax.set_title('검증 20장 점수 — stage1 기준 정렬');grid(ax,'x');fig.tight_layout()
    save(fig,'14_per_image','이미지별 성능','단계별 분포와 어려운 영상을 표시했다. A71의 낮은 점수는 absent Primary Si 규칙과 다른 클래스의 잔여 오류를 함께 고려해야 한다. [S07]')
    mat=100*np.array([r['class_iou'] for r in rows]);fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
    for i in range(20):
        for j in range(4):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>72 else INK)
    ax.set_xticks(range(4),CLASSES);ax.set_yticks(range(20),names);ax.set_title('Stage1: 이미지 × 클래스 IoU (%)');fig.colorbar(im,ax=ax,shrink=.65);fig.tight_layout()
    save(fig,'15_heatmap','이미지·클래스별 IoU','stage1의 80개 IoU를 표시했다. 평균이 숨기는 이미지별·클래스별 차이를 확인할 수 있다. [S07]')
    fig,axs=plt.subplots(1,2,figsize=(12,7),sharey=True)
    for ax,trans,title in zip(axs,['R→stage1','stage1→stage2'],['R → R+E','R+E → R+E+F']):
        by={r['image']:r['delta_pp'] for r in CH['images'] if r['transition']==trans};v=[by[n] for n in names]
        ax.barh(range(20),v,color=[TEAL if x>=0 else RED for x in v]);ax.axvline(0,color=GRAY,lw=.7);ax.set_xlabel('image mIoU 변화 (%p)');ax.set_title(title);grid(ax,'x')
    axs[0].set_yticks(range(20),names);fig.tight_layout()
    save(fig,'16_deltas','이미지별 단계 변화','양수는 개선, 음수는 악화다. 평균 개선이 모든 이미지의 개선을 의미하지 않는다. 두 패널의 가로축 범위는 각각 자동 설정했다. [S07]')
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,trans,title in zip(axs,['R→stage1','stage1→stage2'],['R → R+E','R+E → R+E+F']):
        r=CH['summary'][trans];v=[r['corrected'],r['worsened'],r['wrong_to_wrong']];b=ax.bar(['오류→정답','정답→오류','다른 오류로'],v,color=[TEAL,RED,GRAY]);ax.bar_label(b,labels=[f'{x:,}' for x in v],padding=4,fontsize=8);ax.set_title(title);ax.set_ylim(0,max(v)*1.25);ax.ticklabel_format(axis='y',style='plain');grid(ax)
    fig.tight_layout(rect=[0,.1,1,1]);footer(fig,'픽셀 수 진단이며 image mIoU와 가중 방식이 다름. 전체 픽셀 정확도나 global mIoU로 대체하지 않음.')
    save(fig,'17_pixel_changes','변경 픽셀의 결과','변경된 위치를 정답 교정·새 오류·오류 간 이동으로 구분했다. image mIoU 변화와는 별도 진단이다. [S07]')

def runtime_figures():
    fig,ax=plt.subplots(figsize=(11,4.2));colors={'release':BLUE,'expert':TEAL,'refiner':ORANGE}
    for t in TIME['tasks']:
        if t['gpu'] is None:continue
        color=colors.get(t['family'],GRAY) if t['role']=='train' else '#B9C3D0'
        ax.barh(t['gpu'],(t['end']-t['start'])/3600,left=t['start']/3600,height=.62,color=color,edgecolor='white',lw=.3)
    ax.set_yticks(range(2,8),[f'GPU {g}' for g in range(2,8)]);ax.set_xlabel('실행 시작 후 경과 시간 (시간)');ax.set_title('실제 이벤트 로그의 GPU 작업 배정');grid(ax,'x')
    ax.legend(handles=[Patch(color=BLUE,label='R 학습'),Patch(color=TEAL,label='E 학습'),Patch(color=ORANGE,label='F 학습'),Patch(color='#B9C3D0',label='OOF/valid 추론')],ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.17))
    fig.tight_layout();save(fig,'18_gpu_timeline','GPU 실행 시간선','6대의 RTX 4090에서 작업별 한 GPU를 사용했다. 표기 시간은 작업 점유 경과 시간이며 GPU kernel의 순수 연산 시간이나 DDP 시간은 아니다. [S10]')
    rows=read(PACKAGE/'evidence/refiner_loss.json');fig,ax=plt.subplots(figsize=(8,3.6));ax.plot([r['epoch'] for r in rows],[r['loss'] for r in rows],'o-',color=ORANGE);ax.set(xlabel='Epoch',ylabel='학습 목적함수',title='F의 8-epoch 학습 손실');ax.set_xticks(range(1,9));grid(ax);fig.tight_layout(rect=[0,.12,1,1]);footer(fig,'train70 OOF crop 학습 손실. valid 성능 곡선 또는 epoch 선택 근거가 아님.')
    save(fig,'19_refiner_loss','F 학습 곡선','학습 기록의 평균 loss를 그대로 표시했다. 마지막 epoch8을 고정 사용하며 validation 점수로 epoch를 고르지 않았다. [S06, S10]')

def cases():
    choices=[('20_hard_case','A71_step2_sample1','최저 점수 사례'),('21_best_case','A44_step4_sample3','최고 점수 사례'),('22_refiner_gain',CH['summary']['stage1→stage2']['best']['image'],'F 개선 최대 사례'),('23_stage1_loss',CH['summary']['R→stage1']['worst']['image'],'R 대비 Stage1 손실 최대 사례')]
    for file,name,title in choices:
        image=np.array(Image.open(EXP/f'data/valid/images/{name}_image.png').convert('RGB'));gt=decode(EXP/f'data/valid/masks/{name}_mask.png')
        left,right=('R','stage1') if file=='23_stage1_loss' else ('stage1','stage2')
        folders={'R':RUN/'release_inference','stage1':RESULT,'stage2':RESULT/'stage2'}
        p1=np.array(Image.open(folders[left]/f'predicted_labels/{name}.png'));p2=np.array(Image.open(folders[right]/f'predicted_labels/{name}.png'))
        err=np.full((*gt.shape,3),255,np.uint8);wrong=p1!=gt;err[wrong]=PALETTE[gt[wrong]]
        changed=np.full((*gt.shape,3),255,np.uint8);change=p1!=p2;changed[change]=PALETTE[p2[change]]
        fig,axs=plt.subplots(2,3,figsize=(11,6.7))
        labels={'R':'R (exh14)','stage1':'Stage1 R+E','stage2':'Stage2 R+E+F'}
        for ax,a,t in zip(axs.flat,[image,PALETTE[gt],err,PALETTE[p1],PALETTE[p2],changed],['Original','Ground truth',f'{left} 오류 (정답 색)',labels[left],labels[right],f'{right} 변경 위치 (새 예측 색)']):ax.imshow(a);ax.set_title(t,fontsize=10);ax.axis('off')
        s1=next(r['image_miou'] for r in M[left]['images'] if r['image']==name)*100;s2=next(r['image_miou'] for r in M[right]['images'] if r['image']==name)*100
        fig.suptitle(f'{title} | {name}\n{left} {s1:.6f}% → {right} {s2:.6f}%',fontsize=14)
        fig.subplots_adjust(left=.02,right=.98,bottom=.025,top=.82,wspace=.06,hspace=.24)
        save(fig,file,title,f'{name}의 원본·정답·예측·오류·변경 위치. 흰색 오류 맵은 올바르게 분류된 픽셀이다. 변경 위치 맵은 정오와 무관하게 새 예측 색으로 표시한다. [S02, S07]')

def main():
    for function in tqdm([eda_figures,methods,performance,runtime_figures,cases],desc='Generate report figures',unit='group'):function()
    write(PACKAGE/'figures/catalog.json',CAT)
    print(f'Created {len(CAT)} PNG figures and {len(CAT)} SVG figures.')

if __name__=='__main__':main()
