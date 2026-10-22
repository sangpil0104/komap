import re, sys, zipfile
src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src); x = z.read('word/document.xml').decode()
P = list(re.finditer(r'<w:p[ >].*?</w:p>', x, re.S)); ps = [m.group(0) for m in P]
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
txt = lambda p: ''.join(m.group(2) for m in T.finditer(p)).strip()
def sub(p, old, new):
    ms = [m for m in T.finditer(p) if m.group(2) == old]; assert len(ms) == 1, old[:40]
    m = ms[0]; return p[:m.start(2)] + new + p[m.end(2):]
assert txt(ps[304]).startswith('□ 경계 오류:')
old = [m.group(2) for m in T.finditer(ps[304])][2]
tpl = ps[304]
mk = lambda lab, body: sub(sub(tpl, '경계 오류:', lab), old, body)
ps[304] += mk('상 쌍별 혼동:', ' 전체 오류의 73%가 Al₃Ni와 Al·Eutectic Si 사이에서 발생하고 그중 90%가 경계에 집중됨. Primary Si↔Eutectic Si 혼동(8.4%)만 경계 비율이 54%로 낮아, 같은 Si 상을 입자 단위로 잘못 분류함')
ps[304] += mk('입자 크기:', ' 100픽셀 미만 미세 입자의 재현율이 Al₃Ni 68.3%, Eutectic Si 73.8%로 큰 입자(93~99%)보다 낮고, 각 상에서 놓친 픽셀의 약 40%를 차지')
out, last = [], 0
for i, m in enumerate(P):
    out.append(x[last:m.start()]); out.append(ps[i]); last = m.end()
out.append(x[last:]); x2 = ''.join(out)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist(): o.writestr(it, x2.encode() if it.filename == 'word/document.xml' else z.read(it.filename))
print('ok')
