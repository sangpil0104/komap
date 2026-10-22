import re, sys, zipfile
src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src); x = z.read('word/document.xml').decode()
P = list(re.finditer(r'<w:p[ >].*?</w:p>', x, re.S)); ps = [m.group(0) for m in P]
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
txt = lambda p: ''.join(m.group(2) for m in T.finditer(p)).strip()
def sub(p, old, new):
    ms = [m for m in T.finditer(p) if m.group(2) == old]; assert len(ms) == 1, old[:40]
    m = ms[0]; return p[:m.start(2)] + new + p[m.end(2):]
i = [k for k, p in enumerate(ps) if txt(p).startswith('□ 입자 크기:')]; assert len(i) == 1; i = i[0]
tpl = ps[i]; old = [m.group(2) for m in T.finditer(tpl)][2]
mk = lambda lab, body: sub(sub(tpl, '입자 크기:', lab), old, body)
ps[i] += mk('경계·형상 지표:', ' 경계 3픽셀 폭으로 계산한 Boundary IoU는 69.9%로 mIoU보다 10.9%p 낮음. 10픽셀 이상 입자의 개수·평균 크기 오차는 이미지별 중앙값 6~11%')
ps[i] += mk('검토 영역 선별:', ' R과 E의 판단이 다른 영역은 면적 2.75%이지만 오류율이 38.7%로 전체 평균(5.76%)의 6.7배. E 확신도가 낮은 면적 10%만 검토해도 전체 오류의 62%를 찾음(무작위 선정 시 10%)')
j = [k for k, p in enumerate(ps) if txt(p).startswith('□ 모델 간 불일치를 활용한 검토:')]; assert len(j) == 1; j = j[0]
o2 = [m.group(2) for m in T.finditer(ps[j])][2]
assert o2.endswith(' 불일치가 실제 오류를 얼마나 잘 찾는지 별도로 검증할 계획'), o2
ps[j] = sub(ps[j], o2, o2.replace('. 불일치가 실제 오류를 얼마나 잘 찾는지 별도로 검증할 계획', ''))
out, last = [], 0
for k, m in enumerate(P):
    out.append(x[last:m.start()]); out.append(ps[k]); last = m.end()
out.append(x[last:]); x2 = ''.join(out)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist(): o.writestr(it, x2.encode() if it.filename == 'word/document.xml' else z.read(it.filename))
print('ok')
