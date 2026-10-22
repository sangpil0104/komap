import re, sys, zipfile
src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src); x = z.read('word/document.xml').decode()
P = list(re.finditer(r'<w:p[ >].*?</w:p>', x, re.S)); ps = [m.group(0) for m in P]
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
txt = lambda p: ''.join(m.group(2) for m in T.finditer(p)).strip()
def sub(p, old, new):
    ms = [m for m in T.finditer(p) if m.group(2) == old]; assert len(ms) == 1, old[:40]
    m = ms[0]; return p[:m.start(2)] + new + p[m.end(2):]
def find(pre):
    k = [i for i, p in enumerate(ps) if txt(p).startswith(pre)]; assert len(k) == 1, pre; return k[0]
def body(i): return [m.group(2) for m in T.finditer(ps[i])][2]
a = find('□ 경계 정밀화와 측정 신뢰도:')
ps[a] = sub(ps[a], body(a), ' 경계 학습을 강화하고, Boundary IoU·상 면적비·형상 측정 오차로 개선 효과를 확인할 계획')
ps[find('- 평가 확장:')] = ''
b = find('□ 모델 간 불일치를 활용한 검토:')
ps[b] = sub(ps[b], body(b), ' 판단이 다른 영역을 전문가 검토·추가 라벨링 후보로 제시하고, 수정 시간과 후속 학습 반영 효과를 평가할 계획')
ps[find('- 검토 효율 평가:')] = ''
out, last = [], 0
for k, m in enumerate(P):
    out.append(x[last:m.start()]); out.append(ps[k]); last = m.end()
out.append(x[last:]); x2 = ''.join(out)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist(): o.writestr(it, x2.encode() if it.filename == 'word/document.xml' else z.read(it.filename))
print('ok')
