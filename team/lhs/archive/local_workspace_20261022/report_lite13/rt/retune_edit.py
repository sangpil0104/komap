"""Apply the lite13 retuned (probability re-combination) numbers to the report and spell B4 as EfficientNet-B4."""
import re, sys, zipfile, shutil, io
from pathlib import Path
from PIL import Image
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
H = Path(__file__).resolve().parent.parent
z = zipfile.ZipFile(src)
x = z.read('word/document.xml').decode()
paras = list(re.finditer(r'<w:p[ >].*?</w:p>', x, re.S))
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')

def ptext(p):
    return ''.join(m.group(2) for m in T.finditer(p))

# (paragraph index, old, new); old must sit inside one <w:t>
EDITS = [
    (124, 'E 단독 점수(79.62%)는 R(79.81%)보다 낮지만', 'E 단독 점수(79.59%)는 R(79.80%)보다 낮지만'),
    (124, '80.68%(+0.88%p)', '80.75%(+0.95%p)'),
    (124, 'B4와 HRNet을 반씩 결합', 'EfficientNet-B4 40%·HRNet 60%로 결합'),
    (125, '80.68% → 80.72%(+0.04%p)', '80.75% → 80.78%(+0.03%p)'),
    (125, '80.719~80.721%', '80.778~80.781%'),
    (164, '79.81%', '79.80%'), (165, '76.61%', '76.58%'), (167, '94.16%', '94.18%'), (168, '61.43%', '61.40%'),
    (170, '79.62%', '79.59%'), (171, '75.85%', '75.70%'), (172, '86.58%', '86.61%'), (173, '94.50%', '94.52%'), (174, '61.56%', '61.54%'),
    (176, '80.72%', '80.78%'), (177, '78.18%', '78.25%'), (178, '87.32%', '87.33%'), (179, '94.57%', '94.58%'), (180, '62.82%', '62.96%'),
    (184, '80.72%로, 단일 모델 대비 약 2.3%p 향상', '80.78%로, 단일 모델 대비 약 2.4%p 향상'),
    (195, '(Al₃Ni 15%, Al·Eutectic Si 7.5%)', '(Eutectic Si 7.5%)'),
    (203, 'Al₃Ni (35%)·Al (11.25%) 보조', 'Al₃Ni (50%)·Al (18.75%) 보조'),
    (215, 'E의 50%', 'E의 40%'),
    (219, 'E의 25%', 'E의 42%'), (223, 'E의 25%', 'E의 42%'), (227, 'E의 25%', 'E의 18%'),
    (255, '평균 50%', '평균 40%'), (255, '평균 25%', '평균 42%'), (255, '25%로', '18%로'),
    (263, '80.720984%', '80.779378%'),
    (271, '78.18%', '78.25%'), (272, '89.5%', '89.6%'), (273, '90.5%', '90.4%'),
    (276, '87.32%', '87.33%'), (281, '94.57%', '94.58%'),
    (286, '62.82%', '62.96%'), (287, '76.9%', '77.1%'), (288, '80.7%', '80.8%'), (291, '80.72%', '80.78%'),
    (296, 'A71(53.68%)', 'A71(53.63%)'), (296, '71.58%', '71.51%'),
    (297, '88.7%', '88.4%'), (309, '88.7%', '88.4%'),
    (306, '76.9%', '77.1%'), (306, '80.7%', '80.8%'),
    (307, '31.3%·40.4%', '31.5%·40.8%'), (307, '59.4%·64.9%', '59.4%·64.8%'),
]
new = {i: m.group(0) for i, m in enumerate(paras)}
missing = []
for i, old, rep in EDITS:
    p = new[i]; hits = [m for m in T.finditer(p) if old in m.group(2)]
    if len(hits) != 1:
        missing.append((i, old, ptext(p)[:120])); continue
    m = hits[0]
    assert m.group(2).count(old) == 1, (i, old)
    new[i] = p[:m.start(2)] + m.group(2).replace(old, rep) + p[m.end(2):]
if missing:
    for r in missing: print('MISS', r)
    sys.exit(1)

# B4 -> EfficientNet-B4 wherever the paragraph text does not already say EfficientNet-B4
nb4 = 0
for i, p in new.items():
    full = ptext(p)
    if 'B4' not in full:
        continue
    out, pos, last = [], 0, 0
    for m in T.finditer(p):
        t = m.group(2); start = pos; pos += len(t)
        def sub(mm):
            global nb4
            g = start + mm.start()
            if full[max(0, g - 13):g] == 'EfficientNet-':
                return mm.group(0)
            nb4 += 1
            return 'EfficientNet-B4'
        t2 = re.sub(r'B4', sub, t)
        out.append(p[last:m.start(2)] + t2); last = m.end(2)
    out.append(p[last:]); new[i] = ''.join(out)
print('B4 renamed', nb4)

parts, last = [], 0
for i, m in enumerate(paras):
    parts.append(x[last:m.start()]); parts.append(new[i]); last = m.end()
parts.append(x[last:])
x2 = ''.join(parts)

# figures 2-4: same media slots and the same aspect ratios, so the layout does not move
def fit(path, like, fmt):
    old = Image.open(io.BytesIO(z.read(like)))
    im = Image.open(path).convert('RGB')
    assert abs(im.width / im.height - old.width / old.height) < 0.01, (path, im.size, old.size)
    buf = io.BytesIO(); im.save(buf, fmt, **({'quality': 95} if fmt == 'JPEG' else {})); return buf.getvalue()
media = {'word/media/image2.png': fit(H / 'figure2_retuned.png', 'word/media/image2.png', 'PNG'),
         'word/media/image3.png': fit(H / 'figure3_retuned.png', 'word/media/image3.png', 'PNG'),
         'word/media/image4.jpeg': fit(H / 'figure4_retuned.png', 'word/media/image4.jpeg', 'JPEG')}
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist():
        data = x2.encode() if it.filename == 'word/document.xml' else media.get(it.filename, z.read(it.filename))
        o.writestr(it, data)
print('written', dst)
