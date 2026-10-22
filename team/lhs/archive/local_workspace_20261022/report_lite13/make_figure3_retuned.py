"""Figure 3 with the table-5 model numbers: R class-wise shares, larger (15.5 x 8.3 cm) to fill the space under it in Word."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

O = Path(__file__).resolve().parent
FONT = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
S = 2


def font(n, bold=False):
    return ImageFont.truetype(FONT, int(n * S), index=6 if bold else 0)


def P(*v):
    return tuple(int(round(x * S)) for x in v)


def txt(d, xy, s, size, color='#1b1f24', bold=False, anchor='mm'):
    d.text(P(*xy), s, font=font(size, bold), fill=color, anchor=anchor)


# shares from the lite13 route config (same derivation as make_figures13.py)
ROWS = [('Al₃Ni', [('R3', 50), ('R46', 50)]),
        ('Primary Si', [('R46', 75), ('R7', 25)]),
        ('Al', [('R3', 18.75), ('R7', 81.25)]),
        ('Eutectic Si', [('R1', 7.5), ('R2', 17.5), ('R7', 75)])]
for _, segs in ROWS:
    assert abs(sum(v for _, v in segs) - 100) < 1e-9
M = {'R1': ('#8e98a8', 'R1', 'ResNet34 고해상도'), 'R2': ('#3f4552', 'R2', 'ResNet34 원본 해상도'),
     'R3': ('#4a90e2', 'R3', 'EfficientNet-B4 3배 확대'), 'R46': ('#1f4e9c', 'R4~R6', 'EfficientNet-B4 가로 4000 (3개 평균)'),
     'R7': ('#2e9e5b', 'R7', 'HRNet 드문 상 가중')}
MAIN = ['R3·R4~R6 분담', 'R4~R6 (75%)', 'R7 (81.25%)', 'R7 (75%)']

W, H = 1655, 982
im = Image.new('RGB', P(W, H), 'white'); d = ImageDraw.Draw(im)
x0, x1, xr = 205, 1420, 1540
txt(d, (100, 34), '상', 26, '#5b6b7f', True)
txt(d, ((x0 + x1) / 2, 34), 'R 결합에서 각 모델 확률이 차지하는 비중 (상마다 합 100%)', 26, '#5b6b7f', True)
txt(d, (xr, 34), '주 담당', 26, '#5b6b7f', True)
d.line(P(20, 68, W - 20, 68), fill='#d5dce6', width=S)
for i, (lab, segs) in enumerate(ROWS):
    y = 92 + i * 165; h = 130
    if i % 2:
        d.rectangle(P(20, y - 17, W - 20, y + h + 17), fill='#f6f8fb')
    txt(d, (188, y + h / 2), lab, 30, anchor='rm', bold=True)
    x = x0
    for k, v in segs:
        w = v / 100 * (x1 - x0); col, num, name = M[k]
        d.rectangle(P(x, y, x + w - 3, y + h), fill=col)
        cx = x + w / 2
        if w > 420:
            txt(d, (cx, y + h / 2 - 20), f'{num}  {name}', 28, 'white', True)
            txt(d, (cx, y + h / 2 + 24), f'{v:g}%', 27, 'white')
        else:
            txt(d, (cx, y + h / 2 - 20), num, 27 if w > 110 else 25, 'white', True)
            txt(d, (cx, y + h / 2 + 22), f'{v:g}%', 25 if w > 110 else 22, 'white')
        x += w
    txt(d, (xr, y + h / 2), MAIN[i], 28, '#1b1f24', True)
ya = 92 + 4 * 165 - 14
d.line(P(x0, ya, x1, ya), fill='#1b1f24', width=S)
for t in range(5):
    xt = x0 + t * (x1 - x0) / 4
    d.line(P(xt, ya, xt, ya + 9), fill='#1b1f24', width=S); txt(d, (xt, ya + 32), f'{25 * t}%', 24)
ly = ya + 102
for j, (col, num, name) in enumerate(M.values()):
    r, c = divmod(j, 3); c = 2 if j == 4 else c; cx = x0 + c * 470; cy = ly + r * 62  # R7 under R3: the R4~R6 name is long
    d.rounded_rectangle(P(cx, cy - 17, cx + 44, cy + 17), 5 * S, fill=col)
    txt(d, (cx + 58, cy), num, 26, '#1b1f24', True, 'lm')
    txt(d, (cx + 58 + d.textlength(num, font=font(26, True)) / S + 14, cy), name, 26, '#1b1f24', anchor='lm')
im.save(O / 'figure3_retuned.png', dpi=(300, 300))
print(im.size, 'ratio', im.size[0] / im.size[1])
