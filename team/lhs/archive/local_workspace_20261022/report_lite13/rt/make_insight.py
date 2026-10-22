"""Insert the originality/insight paragraphs (copied with their formatting from the earlier insight version) into the current final report, with retuned numbers."""
import re, sys, zipfile
src, ins, dst = sys.argv[1:4]
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
P = re.compile(r'<w:p[ >].*?</w:p>', re.S)
text = lambda p: ''.join(m.group(2) for m in T.finditer(p))
z = zipfile.ZipFile(src); x = z.read('word/document.xml').decode()
y = zipfile.ZipFile(ins).read('word/document.xml').decode()
yp = P.findall(y)
def take(prefix, reps):
    hit = [p for p in yp if text(p).lstrip().startswith(prefix)]; assert len(hit) == 1, prefix; p = hit[0]
    for old, new in reps:
        ms = [m for m in T.finditer(p) if old in m.group(2)]
        assert len(ms) == 1, (prefix, old, [m.group(2) for m in T.finditer(p)]); m = ms[0]
        p = p[:m.start(2)] + m.group(2).replace(old, new) + p[m.end(2):]
    return p
add = {
    '- 품질 관리:': [take('- 측정 정확도:', []), take('□ 독창성:', [('80.72%', '80.78%')])],
    '- 앙상블 효과:': [take('- 차별점:', [])],
    '□ 추가 경량화:': [take('□ 활용 방안:', [('0.32', '0.31'), ('0.77', '0.74'), ('80.72%', '80.78%'), ('B4 단일', 'EfficientNet-B4 단일')])],
}
for anchor, paras in add.items():
    ms = [m for m in P.finditer(x) if text(m.group(0)).lstrip().startswith(anchor)]
    assert len(ms) == 1, anchor; m = ms[0]
    x = x[:m.end()] + ''.join(paras) + x[m.end():]
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist():
        o.writestr(it, x.encode() if it.filename == 'word/document.xml' else z.read(it.filename))
print('ok')
