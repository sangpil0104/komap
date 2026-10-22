"""lite13 retuned report figures 2 and 4 (E ratio 40/60, retuned test masks), drawn in the style of the figures already in the report (2x pixel scale,
same aspect ratios as the images they replace so the Word layout is unchanged)."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

O = Path(__file__).resolve().parent
SRC = O / 'fig_src'
FONT = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
PAL = np.array([[25, 76, 153], [76, 178, 76], [153, 127, 76], [204, 204, 204]], np.uint8)
S = 2  # pixel scale vs. the original figure


def font(n, bold=False):
    return ImageFont.truetype(FONT, int(n * S), index=6 if bold else 0)


def P(*v):
    return tuple(int(x * S) for x in v)


def txt(d, xy, s, size, color='#1b293c', bold=False, anchor='mm'):
    d.text(P(*xy), s, font=font(size, bold), fill=color, anchor=anchor)


def box(d, rect, fill='#ffffff', outline='#d5dce6', radius=12, width=1.5):
    d.rounded_rectangle(P(*rect), int(radius * S), fill=fill, outline=outline, width=max(1, int(width * S)))


def arrow(d, pts, color='#35485f', width=2):
    pts = [P(*p) for p in pts]
    d.line(pts, fill=color, width=int(width * S), joint='curve')
    (px, py), (x, y) = pts[-2], pts[-1]
    v = np.array([x - px, y - py], float); v /= np.linalg.norm(v)
    a = np.array([x, y]) - v * 9 * S; per = np.array([-v[1], v[0]]) * 5 * S
    d.polygon([(x, y), tuple(a + per), tuple(a - per)], fill=color)


def mask_rgb(name):
    return Image.fromarray(PAL[np.asarray(Image.open(SRC / f'{name}_label13r.png'))])


# ---------------------------------------------------------------- figure 2: structure (1410 x 665)
BLUE, GREEN = '#2563d6', '#0f7a66'
im = Image.new('RGB', P(1410, 665), 'white'); d = ImageDraw.Draw(im)
# R group
box(d, (230, 22, 858, 298), '#f5f7fa', '#d5dce6', 14)
d.ellipse(P(244, 40, 276, 72), fill=BLUE); txt(d, (260, 56), 'R', 17, 'white', True)
txt(d, (292, 56), '상별 결합 앙상블', 21, '#1b293c', True, 'lm'); txt(d, (842, 56), '7개', 17, '#5b6b7f', anchor='rm')
for x, title, cnt, sub in [(242, 'ResNet34', '2개', '기본 확률 담당'), (449, 'EfficientNet-B4', '4개', '밝기 차이에 강함'),
                           (656, 'HRNet-W48', '1개', '얇은 경계에 강함')]:
    box(d, (x, 92, x + 192, 284), 'white', '#d5dce6', 8)
    d.rectangle(P(x, 92, x + 192, 96), fill=BLUE)
    txt(d, (x + 96, 142), title, 19, '#1b293c', True); txt(d, (x + 96, 195), cnt, 26, BLUE, True)
    txt(d, (x + 96, 244), sub, 16, '#4b5a6b')
# E group
box(d, (230, 343, 858, 618), '#f5f7fa', '#d5dce6', 14)
d.ellipse(P(244, 362, 276, 394), fill=GREEN); txt(d, (260, 378), 'E', 17, 'white', True)
txt(d, (292, 378), '전문가 앙상블', 21, '#1b293c', True, 'lm'); txt(d, (842, 378), '5개 · 가로 폭 정규화', 17, '#5b6b7f', anchor='rm')
for x, w, title, cnt, sub in [(242, 296, 'EfficientNet-B4', '2개 · 40%', 'BatchNorm 고정 추가 학습'),
                              (550, 296, 'HRNet-W48', '3개 · 60%', 'fold·전체 학습·밝기 입력 다양화')]:
    box(d, (x, 413, x + w, 604), 'white', '#d5dce6', 8)
    d.rectangle(P(x, 413, x + w, 417), fill=GREEN)
    txt(d, (x + w / 2, 463), title, 19, '#1b293c', True); txt(d, (x + w / 2, 517), cnt, 26, GREEN, True)
    txt(d, (x + w / 2, 566), sub, 16, '#4b5a6b')
# input image + arrows
raw = Image.open(SRC / 'A107_step1_sample3_image.png').convert('RGB'); raw.thumbnail(P(160, 160))
ih = raw.height / S; im.paste(raw, P(24, 322 - ih / 2)); txt(d, (104, 322 + ih / 2 + 22), '입력 이미지', 17, '#1b293c', True)
arrow(d, [(186, 322), (208, 322), (208, 160), (229, 160)]); arrow(d, [(208, 322), (208, 482), (229, 482)])
# flow on the right
for (y1, y2, t1, t2) in [(116, 206, '상별 결합 · 후처리', '상마다 비율 반영'), (299, 389, '덮어쓰기', 'Al₃Ni·Al을 E로 교체'),
                         (483, 573, 'F 보정', 'Al 다시 확인')]:
    box(d, (907, y1, 1134, y2), 'white', '#35485f', 8, 1.5)
    txt(d, (1020, y1 + 32), t1, 19, '#1b293c', True); txt(d, (1020, y1 + 62), t2, 16, '#4b5a6b')
arrow(d, [(860, 160), (905, 160)]); arrow(d, [(1020, 206), (1020, 297)]); arrow(d, [(1020, 389), (1020, 481)])
arrow(d, [(860, 482), (884, 482), (884, 344), (905, 344)], GREEN)
m = mask_rgb('A107_step1_sample3'); m.thumbnail(P(160, 160)); mh = m.height / S; im.paste(m, P(1193, 528 - mh / 2))
arrow(d, [(1134, 528), (1190, 528)]); txt(d, (1273, 528 + mh / 2 + 22), '최종 마스크', 17, '#1b293c', True)
im.save(O / 'figure2_retuned.png', dpi=(300, 300))

# ---------------------------------------------------------------- figure 4: test predictions (749 x 602)
im = Image.new('RGB', P(749, 602), 'white'); d = ImageDraw.Draw(im)
y = 4
for name in ['A107_step1_sample3', 'A51_step3_sample3']:
    raw = Image.open(SRC / f'{name}_image.png').convert('RGB'); m = mask_rgb(name)
    w = 368; h = 258
    im.paste(raw.resize(P(w, h), Image.Resampling.LANCZOS), P(4, y))
    im.paste(m.resize(P(w, h), Image.Resampling.NEAREST), P(377, y))
    y += h + 6
labs = ['Al₃Ni', 'Primary Si', 'Al', 'Eutectic Si']
widths = [36 + 10 + d.textlength(l, font=font(19)) / S for l in labs]; gap = 40
x = (749 - sum(widths) - gap * 3) / 2
for lab, col, w in zip(labs, PAL, widths):
    d.rectangle(P(x, 565, x + 36, 588), fill=tuple(int(c) for c in col), outline='#888888')
    txt(d, (x + 46, 577), lab, 19, '#1b1f24', anchor='lm'); x += w + gap
im.save(O / 'figure4_retuned.png', dpi=(300, 300))
print('figures written')
