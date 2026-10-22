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
for x,title,count,lines in [(240,'ResNet34','2개',[]), (550,'EfficientNet-B4','6개',[]), (860,'HRNet-W48','1개',[])]:
 box(d,(x,110,x+295,370));d.rectangle((x+10,111,x+285,117),fill='#2864cc');txt(d,(x+148,154),title,37,bold=True);txt(d,(x+148,255),count,47,'#2462d0',True)
 for j,s in enumerate(lines):txt(d,(x+148,267+j*37),s,32,'#516881')
box(d,(220,455,1180,880),'#f0f8f6','#adcec8');txt(d,(255,497),'E  전문가 결합 앙상블',44,'#08796e',True,'lm');txt(d,(1125,497),'6개',40,'#3e746e')
for x,title,percent,lines in [(240,'EfficientNet-B4','50%',['2개']), (550,'HRNet-W48','25%',['2개']), (860,'HRNet-W48','25%',['2개'])]:
 box(d,(x,545,x+295,795));d.rectangle((x+10,546,x+285,552),fill='#087e71');txt(d,(x+148,589),title,36,bold=True);txt(d,(x+148,648),percent,45,'#087e71',True)
 for j,s in enumerate(lines):txt(d,(x+148,706+j*38),s,32,'#4b6d69')
txt(d,(700,840),'그룹별 평균 확률을 위 비율로 결합',31,'#46665e')
# Input image.
raw=Image.open(O/'A107_step1_sample3_image.png').convert('RGB');raw.thumbnail((170,210));im.paste(raw,(25,375));txt(d,(110,623),'입력 이미지',30,bold=True)
arrow(d,[(195,455),(208,455),(208,200),(220,200)]);arrow(d,[(208,455),(208,650),(220,650)],'#0b877b')
box(d,(1240,110,1550,280),'#ffffff','#354b67');txt(d,(1395,167),'상별 결합·후처리',38,bold=True);txt(d,(1395,226),'R 기본 라벨',34,'#566c86');arrow(d,[(1180,210),(1240,210)])
box(d,(1240,420,1550,600),'#ffffff','#354b67');txt(d,(1395,465),'E 덮어쓰기',40,bold=True);txt(d,(1395,518),'E가 Al₃Ni·Al이면',32,'#566c86');txt(d,(1395,555),'R 라벨을 교체',32,'#566c86');arrow(d,[(1395,280),(1395,420)]);arrow(d,[(1180,620),(1205,620),(1205,520),(1240,520)],'#0b877b')
box(d,(1240,760,1550,975),'#f8f3fd','#9d7dba');txt(d,(1395,809),'F 잔차 보정기',39,bold=True);txt(d,(1395,859),'1개',33,'#716382');txt(d,(1395,908),'입력: E log 확률 + 밝기',28,'#716382');arrow(d,[(1180,805),(1210,805),(1210,875),(1240,875)],'#9367b0');txt(d,(1170,927),'원본 밝기도 입력',26,'#716382',anchor='rm')
# Final merge explicitly keeps stage1 except where F says Al.
box(d,(1590,640,1780,840),'#f4f8fc','#354b67');txt(d,(1685,683),'최종 합성',35,bold=True);txt(d,(1685,733),'F가 Al → Al',29,'#566c86');txt(d,(1685,781),'그 외 → 1단계',27,'#566c86');arrow(d,[(1550,510),(1685,510),(1685,640)]);arrow(d,[(1550,865),(1569,865),(1569,756),(1590,756)],'#9367b0')
mask=Image.fromarray(PAL[np.asarray(Image.open(O/'A107_step1_sample3.png'))]);mask.thumbnail((170,140));im.paste(mask,(1600,864));txt(d,(1685,1020),'최종 마스크',31,bold=True)
txt(d,(720,965),'R 9개 + E 6개 + F 1개 = 가중치 16개',42,bold=True);save(im,'figure2_lite16_N3_structure_public.png')
