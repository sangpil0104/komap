"""Create standalone table images from the same cells used in the manuscript."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[1]
FONT=ROOT/'source/NanumGothic-Regular.ttf'
normal=ImageFont.truetype(str(FONT),32);heading=ImageFont.truetype(str(FONT),40)
blocks=json.loads((ROOT/'source/manuscript.json').read_text())['blocks']
catalog=json.loads((ROOT/'tables/catalog.json').read_text())
measure=ImageDraw.Draw(Image.new('RGB',(1,1)))
def wrap(text,width,font=normal):
    lines=[]
    for raw in str(text).splitlines() or ['']:
        line=''
        for char in raw:
            if measure.textlength(line+char,font=font)>width and line:lines.append(line);line=char
            else:line+=char
        lines.append(line)
    return lines
tables=[b for b in blocks if b['type']=='table']
for i,(table,row) in enumerate(tqdm(list(zip(tables,catalog)),desc='Render table images',unit='table'),1):
    n=len(table['headers']);width=2600;margin=55;usable=width-margin*2
    if n==2:ratios=[.24,.76]
    elif n==3:ratios=[.3,.28,.42]
    elif n>=5:ratios=[.24]+[.76/(n-1)]*(n-1)
    else:ratios=[1/n]*n
    widths=[int(usable*r) for r in ratios];widths[-1]=usable-sum(widths[:-1])
    rows=[]
    for cells in [table['headers'],*table['rows']]:
        lines=[wrap(t,w-34) for t,w in zip(cells,widths)];h=max(len(t) for t in lines)*45+30;rows.append((lines,h))
    note=wrap(table.get('note',''),usable);height=155+sum(h for _,h in rows)+len(note)*43+60
    im=Image.new('RGB',(width,height),'white');d=ImageDraw.Draw(im)
    d.text((margin,40),f'표 {i}. {table["title"]}',font=heading,fill='#20324A')
    y=135
    for j,(lines,h) in enumerate(rows):
        x=margin
        for w,cell in zip(widths,lines):
            d.rectangle((x,y,x+w,y+h),fill='#E8EFF8' if j==0 else ('#F7F9FC' if j%2==0 else 'white'),outline='#B7C4D4',width=2)
            for k,line in enumerate(cell):d.text((x+17,y+15+k*45),line,font=normal,fill='#20324A')
            x+=w
        y+=h
    for k,line in enumerate(note):d.text((margin,y+20+k*43),line,font=normal,fill='#526478')
    name=Path(row['file']).with_suffix('.png');im.save(ROOT/name,dpi=(300,300));row['image']=str(name)
(ROOT/'tables/catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
lines=['# 표 목록','', '| 번호 | 제목 | 편집용 CSV | 표 이미지 |','|---|---|---|---|']
for r in catalog:
    stem=Path(r['file']).stem;lines.append(f'| {r["number"]} | [{r["title"]}]({stem}.md) | [CSV]({stem}.csv) | [PNG]({stem}.png) |')
(ROOT/'tables/표_목록.md').write_text('\n'.join(lines)+'\n')
lines=['# 본문 그림 목록','', '| 본문 번호 | 제목 | PNG | SVG |','|---|---|---|---|']
for i,b in enumerate([b for b in blocks if b['type']=='figure'],1):
    lines.append(f'| {i} | {b["title"]} | [PNG]({Path(b["file"]).name}) | [SVG]({Path(b["svg"]).name}) |')
(ROOT/'figures/그림_목록.md').write_text('\n'.join(lines)+'\n')
