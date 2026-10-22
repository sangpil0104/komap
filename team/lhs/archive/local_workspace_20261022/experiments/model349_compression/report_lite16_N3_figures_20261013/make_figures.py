from pathlib import Path
import base64,html,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;W=O.parents[2];FONT='/System/Library/Fonts/AppleSDGothicNeo.ttc';PAL=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8)
def font(n,bold=False):return ImageFont.truetype(FONT,n,index=6 if bold else 0)
def txt(d,xy,s,size=38,color='#1b293c',bold=False,anchor='mm'):d.text(xy,s,font=font(size,bold),fill=color,anchor=anchor)
def box(d,rect,fill='#ffffff',outline='#9eb2ca',radius=15,width=2):d.rounded_rectangle(rect,radius,fill=fill,outline=outline,width=width)
def arrow(d,points,color='#35485f',width=4):
 d.line(points,fill=color,width=width,joint='curve');x,y=points[-1];px,py=points[-2];v=np.array([x-px,y-py],float);v/=np.linalg.norm(v);a=np.array([x,y])-v*17;per=np.array([-v[1],v[0]])*8;d.polygon([tuple([x,y]),tuple(a+per),tuple(a-per)],fill=color)
def save(im,name):im.save(O/name,dpi=(300,300))
# Figure 2: R and E feed the first-stage result; F receives E probabilities and original brightness.
im=Image.new('RGB',(1800,1050),'white');d=ImageDraw.Draw(im)
box(d,(220,25,1180,395),'#f2f6fc','#b7c7df');txt(d,(255,66),'R  상별 결합 앙상블',44,'#225dc9',True,'lm');txt(d,(1125,66),'9개',40,'#486381')
for x,title,count,lines in [(240,'ResNet34','2개',['restart61','model269']), (550,'EfficientNet-B4','6개',['322 fold3','326 fold1·2','329 fold1·3·4']), (860,'HRNet-W48','1개',['HR347','336·347 역할 공유'])]:
 box(d,(x,110,x+295,370));d.rectangle((x+10,111,x+285,117),fill='#2864cc');txt(d,(x+148,154),title,37,bold=True);txt(d,(x+148,213),count,47,'#2462d0',True)
 for j,s in enumerate(lines):txt(d,(x+148,267+j*37),s,32,'#516881')
box(d,(220,455,1180,880),'#f0f8f6','#adcec8');txt(d,(255,497),'E  전문가 결합 앙상블',44,'#08796e',True,'lm');txt(d,(1125,497),'6개',40,'#3e746e')
for x,title,percent,lines in [(240,'B4ws 2개','50%',['fold1 · fold3','BN 고정 추가학습']), (550,'N3 + HR392','25%',['N3 seed414','HR392 seed1 fold4']), (860,'HR392 2개','25%',['seed2 fold1','seed2 fold4'])]:
 box(d,(x,545,x+295,795));d.rectangle((x+10,546,x+285,552),fill='#087e71');txt(d,(x+148,589),title,36,bold=True);txt(d,(x+148,648),percent,45,'#087e71',True)
 for j,s in enumerate(lines):txt(d,(x+148,706+j*38),s,32,'#4b6d69')
txt(d,(700,840),'N3: 원본 밝기·Z·CDF 입력  /  HR392: 폭 정규화',31,'#46665e')
# Input image.
raw=Image.open(O/'A107_step1_sample3_image.png').convert('RGB');raw.thumbnail((170,210));im.paste(raw,(25,375));txt(d,(110,623),'입력 이미지',30,bold=True)
arrow(d,[(195,455),(208,455),(208,200),(220,200)]);arrow(d,[(208,455),(208,650),(220,650)],'#0b877b')
box(d,(1240,110,1550,280),'#ffffff','#354b67');txt(d,(1395,167),'상별 결합·후처리',38,bold=True);txt(d,(1395,226),'R 기본 라벨',34,'#566c86');arrow(d,[(1180,210),(1240,210)])
box(d,(1240,420,1550,600),'#ffffff','#354b67');txt(d,(1395,465),'E 덮어쓰기',40,bold=True);txt(d,(1395,518),'E가 Al₃Ni·Al이면',32,'#566c86');txt(d,(1395,555),'R 라벨을 교체',32,'#566c86');arrow(d,[(1395,280),(1395,420)]);arrow(d,[(1180,620),(1205,620),(1205,520),(1240,520)],'#0b877b')
box(d,(1240,760,1550,975),'#f8f3fd','#9d7dba');txt(d,(1395,809),'F 잔차 보정기',39,bold=True);txt(d,(1395,859),'1개 · epoch8',33,'#716382');txt(d,(1395,908),'입력: E log 확률 + 밝기',28,'#716382');arrow(d,[(1180,805),(1210,805),(1210,875),(1240,875)],'#9367b0');txt(d,(1170,927),'원본 밝기도 입력',26,'#716382',anchor='rm')
# Final merge explicitly keeps stage1 except where F says Al.
box(d,(1590,640,1780,840),'#f4f8fc','#354b67');txt(d,(1685,683),'최종 합성',35,bold=True);txt(d,(1685,733),'F가 Al → Al',29,'#566c86');txt(d,(1685,781),'그 외 → 1단계',27,'#566c86');arrow(d,[(1550,510),(1685,510),(1685,640)]);arrow(d,[(1550,865),(1569,865),(1569,756),(1590,756)],'#9367b0')
mask=Image.fromarray(PAL[np.asarray(Image.open(O/'A107_step1_sample3.png'))]);mask.thumbnail((170,140));im.paste(mask,(1600,864));txt(d,(1685,1020),'최종 마스크',31,bold=True)
txt(d,(720,965),'R 9개 + E 6개 + F 1개 = 가중치 16개',42,bold=True);txt(d,(720,1013),'N3는 기존 HR392 seed1 fold2 한 자리를 교체',32,'#52677f');save(im,'figure2_lite16_N3_structure.png')
# Figure 3: exact configured coefficients, not a supposed fixed final share.
im=Image.new('RGB',(1800,1050),'white');d=ImageDraw.Draw(im);txt(d,(900,55),'R의 클래스별 순차 교체 계수',50,bold=True);txt(d,(900,112),'restart61 기본 확률에서 시작해 아래 계수를 적용',36,'#52677f')
xs=[40,265,775,1030,1285,1540,1780];ys=[155,235,345,455,565,675,785,895]
head=['단계','사용 모델','Al₃Ni','Primary Si','Al','Eutectic Si'];rows=[('기본 결합','B4-326 (fold1·2)',[.7,1,0,0]),('기본 결합','B4-322 (fold3)',[0,0,.6,0]),('기본 결합','model269',[0,0,0,.7]),('추가 결합 1','B4-329 (fold1·3·4)',[.5,1,0,0]),('추가 결합 2','HR347 (336 역할)',[0,.25,.75,.75]),('추가 결합 3','HR347 (347 역할)',[0,0,.25,0])]
for j,h in enumerate(head):box(d,(xs[j],ys[0],xs[j+1]-4,ys[1]-4),'#eaf0f8','#d0dbe8',8);txt(d,((xs[j]+xs[j+1])/2,(ys[0]+ys[1])/2),h,36,bold=True)
for i,(step,model,values) in enumerate(rows):
 y1,y2=ys[i+1],ys[i+2]
 for j,text in enumerate([step,model]):box(d,(xs[j],y1,xs[j+1]-4,y2-4),'#f7f9fc','#d9e2eb',7);txt(d,((xs[j]+xs[j+1])/2,(y1+y2)/2),text,34,bold=j==1)
 for j,v in enumerate(values):
  color=tuple(int(a*(1-v)+b*v) for a,b in zip((245,247,250),(38,96,165)));box(d,(xs[j+2],y1,xs[j+3]-4,y2-4),color,'#ffffff',7);txt(d,((xs[j+2]+xs[j+3])/2,(y1+y2)/2),'—' if v==0 else f'{v:.2f}',40,'#657589' if v==0 else ('white' if v>=.5 else '#233f5c'),True)
txt(d,(900,948),'각 열'+'의 합은 1이 아닙니다. 결합 후 네 상의 확률 합을 다시 1로 정규화합니다.',33,'#52677f');txt(d,(900,1000),'교체 계수이며 최종 성능 기여도가 아닙니다. HR347 두 행은 같은 가중치를 공유합니다.',31,'#52677f');save(im,'figure3_lite16_N3_R_coefficients.png')
# Figure 4: actual fixed seed414 test output, same two images as the document.
WID=2100;panel=1000;margin=30;gap=40;heights=[];pairs=[]
for name in ['A107_step1_sample3','A51_step3_sample3']:
 raw=Image.open(O/f'{name}_image.png').convert('RGB');pred=Image.fromarray(PAL[np.asarray(Image.open(O/f'{name}.png'))]);assert pred.size==raw.size;h=round(raw.height*panel/raw.width);pairs.append((name,raw.resize((panel,h),Image.Resampling.LANCZOS),pred.resize((panel,h),Image.Resampling.NEAREST)));heights.append(h)
im=Image.new('RGB',(WID,160+sum(heights)+160),'white');d=ImageDraw.Draw(im);txt(d,(530,40),'원본 이미지',48,bold=True);txt(d,(1570,40),'lite16-N3 예측',48,bold=True);y=100
for name,raw,pred in pairs:
 txt(d,(35,y-18),name,32,'#3a4c62',anchor='lm');im.paste(raw,(margin,y));im.paste(pred,(margin+panel+gap,y));y+=raw.height+65
for j,(label,col) in enumerate(zip(['Al₃Ni','Primary Si','Al','Eutectic Si'],PAL)):
 x=210+j*450;d.rectangle((x,y-5,x+45,y+30),fill=tuple(col));txt(d,(x+70,y+12),label,37,anchor='lm')
save(im,'figure4_lite16_N3_test_predictions.png')
print('3 figures generated')
