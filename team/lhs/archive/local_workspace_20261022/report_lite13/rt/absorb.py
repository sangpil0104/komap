import re, sys, zipfile
src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src); x = z.read('word/document.xml').decode()
P = list(re.finditer(r'<w:p[ >].*?</w:p>', x, re.S)); ps = [m.group(0) for m in P]
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
txt = lambda p: ''.join(m.group(2) for m in T.finditer(p)).strip()
def find(pre):
    k = [i for i, p in enumerate(ps) if txt(p).startswith(pre)]; assert len(k) == 1, pre; return k[0]
def set_body(p, new, keep=2):
    ms = list(T.finditer(p)); assert len(ms) > keep
    for j in range(len(ms) - 1, keep - 1, -1):
        m = ms[j]; p = p[:m.start(2)] + (new if j == keep else '') + p[m.end(2):]
    return p
def append(i, tail):
    ps[i] = set_body(ps[i], ''.join(m.group(2) for m in list(T.finditer(ps[i]))[2:]) + tail)
r = find('- 검토 효율화:')
assert '(4-2에서 검증 계획)' in txt(ps[r])
ps[r] = set_body(ps[r], ''.join(m.group(2) for m in list(T.finditer(ps[r]))[2:]).replace('(4-2에서 검증 계획)', '(4-1에서 효과 확인)'))
ps[find('4-1. 추가 분석 및 인사이트')] = ''
append(find('□ 입자 크기:'), ' → 미세 입자 중심의 학습 샘플링 필요')
append(find('□ 경계·형상 지표:'), ' → 경계 학습을 강화하고 별도 시편에서 Boundary IoU·형상 오차로 효과 확인 필요')
s = find('□ 검토 영역 선별:')
append(s, ' → 전문가 검토·추가 라벨링 후보 선정에 활용 가능하며, 수정 시간 단축 효과 확인 필요')
k = find('□ 경량화:')
ps[s] += set_body(ps[k], ' 가로 4000픽셀 EfficientNet-B4 3개를 1개로 증류하면 가중치 2개 축소 가능. 기존 시도는 실패해 학습 방식 개선과 추론 시간 평가 필요')
for pre in ('4-2. 향후 연구', '□ 경계 정밀화와 측정 신뢰도:', '□ 모델 간 불일치를 활용한 검토:', '□ 경량화:', '□ 확장 기능 검증:'):
    ps[find(pre)] = ''
out, last = [], 0
for i, m in enumerate(P):
    out.append(x[last:m.start()]); out.append(ps[i]); last = m.end()
out.append(x[last:]); x2 = ''.join(out)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist(): o.writestr(it, x2.encode() if it.filename == 'word/document.xml' else z.read(it.filename))
print('ok')
