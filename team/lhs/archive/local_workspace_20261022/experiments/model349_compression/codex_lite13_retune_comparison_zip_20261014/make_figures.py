from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
D=Path(__file__).resolve().parent
FONT='/System/Library/Fonts/AppleSDGothicNeo.ttc'
def font(n):return ImageFont.truetype(FONT,n,index=0)
v=json.loads((D/'valid_comparison.json').read_text());t=json.loads((D/'test_comparison.json').read_text());pal=np.array(v['palette'],np.uint8)
BLUE='#1458b0';RED='#bc3037';GREEN='#00866b';BG='#f4f7fb';TXT='#15283f'
labels=['Al₃Ni','Primary Si','Al','Eutectic Si']
def text(draw,pos,s,size=22,fill=TXT):draw.text(pos,s,font=font(size),fill=fill)
def boundary(a):
 q=np.zeros(a.shape,bool);q[1:]|=a[1:]!=a[:-1];q[:,1:]|=a[:,1:]!=a[:,:-1];return q

def panel_maps(raw,b,a,gt=None):
 ch=b!=a;gray=np.asarray(raw.convert('L'));dark=np.repeat((gray*.42).astype(np.uint8)[:,:,None],3,2);amplified=np.asarray(Image.fromarray((ch*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)))>0
 overlay=dark.copy();overlay[amplified]=[240,90,190];overlay[ch]=[255,245,30]
 old=Image.fromarray(pal[b]);new=Image.fromarray(pal[a]);maps=[raw]
 if gt is not None:maps.append(Image.fromarray(pal[gt]))
 maps += [old,new,Image.fromarray(overlay)]
 if gt is not None:
  err=dark.copy();gain=(b!=gt)&(a==gt);loss=(b==gt)&(a!=gt);other=ch&~gain&~loss;err[other]=[155,110,235];err[gain]=[35,225,140];err[loss]=[255,65,65];maps.append(Image.fromarray(err))
 else:
  dest=np.full((*b.shape,3),245,np.uint8);dest[ch]=pal[a[ch]];maps.append(Image.fromarray(dest))
 return maps,ch

def draw_row(name,split,metrics,maps,box=None):
 W=360;n=len(maps);H=round(maps[0].height*W/maps[0].width) if box is None else 360
 canvas=Image.new('RGB',(W*n,H+172),BG);dr=ImageDraw.Draw(canvas)
 heading=f'{name}  |  '+('valid20' if split=='valid' else 'test10')
 if box:heading+='  |  변화 집중 영역 확대'
 text(dr,(16,12),heading,30)
 if split=='valid':subtitle=f"mIoU  lite13 {metrics['lite13_miou_percent']:.4f}% → 재조정 {metrics['retuned_miou_percent']:.4f}%   /   증감 {metrics['delta_pp']:+.4f}%p"
 else:subtitle=f"변경 {metrics['changed_pixels']:,}픽셀 ({metrics['changed_percent']:.4f}%)  /  test 정답 없음: 정확도 우열 평가 불가"
 text(dr,(16,53),subtitle,24)
 headings=['입력','정답','lite13','재조정','변경 위치 강조','정답 대비 변화'] if split=='valid' else ['입력','lite13','재조정','변경 위치 강조','변경 픽셀의 새 클래스','']
 # test has five panels; the last blank heading is unused.
 for j,m in enumerate(maps):
  text(dr,(j*W+12,94),headings[j],24)
  if box:m=m.crop(box)
  canvas.paste(m.resize((W,H),Image.Resampling.NEAREST if j else Image.Resampling.LANCZOS),(j*W,130))
 legend='변경 강조: 노랑=실제 변경, 분홍=가시성을 위해 주변 1픽셀 확대'
 if split=='valid':legend+='   |   초록=오류 복구, 빨강=새 오류, 보라=오류 간 변경'
 text(dr,(12,H+139),legend,19)
 return canvas

def boxes(ch):
 # Highest changed-pixel density in fixed 256px windows, then a nonoverlapping second window.
 from scipy.ndimage import uniform_filter
 rest=ch.copy();h,w=ch.shape;sz=min(256,h,w);out=[]
 for _ in range(2):
  den=uniform_filter(rest.astype(np.float32),size=min(128,sz));cy,cx=np.unravel_index(den.argmax(),den.shape)
  x=max(0,min(w-sz,int(cx)-sz//2));y=max(0,min(h-sz,int(cy)-sz//2));out.append((x,y,x+sz,y+sz));rest[max(0,y-20):min(h,y+sz+20),max(0,x-20):min(w,x+sz+20)]=False
 return out
for split,rows in [('valid',v['rows']),('test',t['rows'])]:
 out=D/f'images/{split}';out.mkdir(parents=True,exist_ok=True)
 for row in rows:
  name=row['image'];a0=D/f'assets/{split}';raw=Image.open(a0/f'inputs/{name}.png').convert('RGB');b=np.array(Image.open(a0/f'original_labels/{name}.png'));a=np.array(Image.open(a0/f'retuned_labels/{name}.png'));gt=np.array(Image.open(a0/f'gt_labels/{name}.png')) if split=='valid' else None
  maps,ch=panel_maps(raw,b,a,gt);draw_row(name,split,row,maps).save(out/f'{name}_full.png')
  for i,box in enumerate(boxes(ch),1):draw_row(name,split,row,maps,box).save(out/f'{name}_zoom{i}.png')
  print(split,name,flush=True)
# Numerical summaries, all image rows, image-weighted mean IoU.
can=Image.new('RGB',(1960,1280),'white');dr=ImageDraw.Draw(can)
text(dr,(28,20),'valid20 이미지별 mIoU 증감 — lite13 vs 재조정',38)
text(dr,(28,76),f"20장 평균: {v['baseline_percent']:.6f}% → {v['retuned_percent']:.6f}%  ({v['delta_pp']:+.6f}%p)   /   12장 상승·8장 하락",27)
text(dr,(28,121),'동일 13개 가중치 사용. valid에서 계수를 선택한 개발 점수이며 독립 일반화 성능이 아닙니다.',24)
cols=[(28,'이미지'),(355,'lite13 %'),(520,'재조정 %'),(710,'증감 %p'),(910,'Al₃Ni'),(1090,'Primary Si'),(1280,'Al'),(1460,'Eutectic Si'),(1690,'복구 / 새 오류 px')]
for x,s in cols:text(dr,(x,177),s,24)
for i,r in enumerate(v['rows']):
 y=224+i*45
 if i%2==0:dr.rectangle((16,y-4,1940,y+36),fill=BG)
 vals=[r['image'],f"{r['lite13_miou_percent']:.4f}",f"{r['retuned_miou_percent']:.4f}",f"{r['delta_pp']:+.4f}"]+[f'{q:+.4f}' for q in r['class_delta_pp']]+[f"{r['corrected_pixels']:,} / {r['new_error_pixels']:,}"]
 for j,((x,_),val) in enumerate(zip(cols,vals)):text(dr,(x,y),val,21,fill=(GREEN if (r['delta_pp'] if j==3 else r['class_delta_pp'][j-4])>=0 else RED) if 3<=j<=7 else TXT)
text(dr,(28,1160),'클래스 열은 각 클래스 IoU의 증감(%p)입니다. 빈 union은 0, 이미지별 4클래스 평균 후 20장 평균.',23)
text(dr,(28,1200),'그림: 입력·정답·lite13·재조정·변경 강조·오류 복구/새 오류. 각 이미지 확대 2장 포함.',23)
(D/'images').mkdir(exist_ok=True);can.save(D/'images/00_valid_summary.png')
can=Image.new('RGB',(1660,850),'white');dr=ImageDraw.Draw(can)
text(dr,(28,20),'test10 예측 변경 — lite13 vs 재조정',38)
text(dr,(28,80),f"총 {t['changed_pixels']:,} / {t['total_pixels']:,}픽셀 ({t['changed_percent']:.4f}%) 변경",29)
text(dr,(28,125),f"변경의 {t['within_2px_old_boundary_share_percent']:.2f}%가 기존 경계 2픽셀 이내. test 정답이 없어 정확도 판단 불가.",24)
cols=[(28,'이미지'),(380,'변경 픽셀'),(600,'변경률 %'),(810,'경계 2px 이내 %'),(1080,'최대 변경 영역 px'),(1400,'Al 면적 Δ%p')]
for x,s in cols:text(dr,(x,185),s,23)
for i,r in enumerate(t['rows']):
 y=238+i*47
 if i%2==0:dr.rectangle((16,y-5,1644,y+38),fill=BG)
 vals=[r['image'],f"{r['changed_pixels']:,}",f"{r['changed_percent']:.4f}",f"{r['within_2px_share_percent']:.2f}",str(r['largest_changed_component_pixels']),f"{r['class_area_delta_pp']['Al']:+.4f}"]
 for (x,_),val in zip(cols,vals):text(dr,(x,y),val,23)
text(dr,(28,765),'최대 변경 영역: 변경 마스크를 연결 성분으로 나눴을 때 가장 큰 영역의 픽셀 수.',22)
can.save(D/'images/01_test_summary.png')
# Plain text tables and complete machine-readable metrics.
lines=['이미지\tlite13 mIoU (%)\t재조정 mIoU (%)\t차이 (%p)\t복구 픽셀\t새 오류 픽셀']
for r in v['rows']:lines.append(f"{r['image']}\t{r['lite13_miou_percent']:.8f}\t{r['retuned_miou_percent']:.8f}\t{r['delta_pp']:+.8f}\t{r['corrected_pixels']}\t{r['new_error_pixels']}")
(D/'valid_image_scores.txt').write_text('\n'.join(lines)+'\n')
readme=f'''lite13 vs 계수 재조정 비교 — 2026-10-14

먼저 images/00_valid_summary.png, images/01_test_summary.png를 확인하세요.
images/valid: 20장 각각 전체 비교 1장 + 변화 집중 확대 2장.
images/test: 10장 각각 전체 비교 1장 + 변화 집중 확대 2장.

valid 그림: 입력 / 정답 / lite13 / 재조정 / 변경 위치 강조 / 오류 변화.
test 그림: 입력 / lite13 / 재조정 / 변경 위치 강조 / 변경된 픽셀의 새 클래스.
변경 강조: 노랑은 실제 변경, 분홍은 가시성을 위해 변경 주변 1픽셀만 확대 표시.
오류 변화: 초록=기존 오류 복구, 빨강=새 오류, 보라=오류 간 라벨 변경.
클래스 색상은 assets의 원본 팔레트를 그대로 사용합니다: {dict(zip(labels,v['palette']))}

valid20 평균 {v['baseline_percent']:.8f}% -> {v['retuned_percent']:.8f}% ({v['delta_pp']:+.8f}%p).
test10 변경 {t['changed_pixels']:,}픽셀 ({t['changed_percent']:.6f}%).
수치 전체: valid_comparison.json / test_comparison.json / valid_image_scores.txt.
assets에는 원본 입력, 모델별 라벨 마스크, valid 정답이 있습니다. test 정답은 없습니다.

양쪽 모두 13개 동일 모델 가중치. 재조정 설정은 valid20에서 선택한 retune_full로 고정:
{json.dumps(v['state'],ensure_ascii=False,indent=2)}

측정: 각 valid 이미지의 4개 클래스 IoU를 평균한 뒤 20장 평균(빈 union=0).
valid20은 반복 개발 및 계수 선택에 사용됐으므로 일반화 성능 향상으로 주장하지 않습니다.
test10에는 정답이 없으므로 변경량/시각 비교만 제공합니다. test 보고 후 추가 계수 선택은 하지 않았습니다.
저장 확률 CPU 재결합입니다. 원래 lite13 valid20/test10 마스크 exact를 확인했습니다.
valid 재조정 점수는 기존 retune_full과 일치합니다. 새 GPU 원본 추론 검증은 하지 않았습니다.
test 기존 E는 분해·재구성 차이 0; B4 비교에는 선택 실험과 동일 FP16 fold 캐시 평균을 사용합니다.
해당 B4 캐시는 원래 fresh 가족 확률과 최대 0.00048828 차이가 있을 수 있습니다.
기존 모델/가중치/릴리스와 GPU 시드 학습은 변경하지 않았습니다.
'''
(D/'README.txt').write_text(readme)
# Validate transferred input assets against server checksum receipt.
checks=json.loads((D/'asset_checksums.json').read_text())
assert all(hashlib.sha256((D/p).read_bytes()).hexdigest()==h for p,h in checks.items())
assert len(list((D/'images/valid').glob('*.png')))==60
assert len(list((D/'images/test').glob('*.png')))==30
print('FIGURES_COMPLETE',flush=True)
