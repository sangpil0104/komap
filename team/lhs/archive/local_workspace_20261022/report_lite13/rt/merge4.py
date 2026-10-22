"""Section 4: findings into 4-1 (추가 분석 및 인사이트), plans 4-2+4-3 merged into 4-2 향후 연구."""
import re, sys, zipfile
src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src); x = z.read('word/document.xml').decode()
P = list(re.finditer(r'<w:p[ >].*?</w:p>', x, re.S)); ps = [m.group(0) for m in P]
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
txt = lambda p: ''.join(m.group(2) for m in T.finditer(p)).strip()
def sub(p, old, new):
    ms = [m for m in T.finditer(p) if m.group(2) == old]; assert len(ms) == 1, old[:40]
    m = ms[0]; return p[:m.start(2)] + new + p[m.end(2):]
exp = {301: '4-1. 추가 분석', 304: '4-2. 인사이트 및 발전 가능성', 309: '4-3. 향후 검증 및 경량화'}
for i, t in exp.items(): assert txt(ps[i]) == t, (i, txt(ps[i]))
assert txt(ps[310]).startswith('□ 확장 기능 검증') and txt(ps[311]).startswith('□ 추가 경량화')
boundary = sub(sub(ps[302], '상별 오류:', '경계 오류:'),
    ' Al은 정답의 97.9%를 찾지만 Eutectic Si는 77.1%만 찾고 예측의 80.8%만 맞아 놓침·과잉 예측이 모두 많음(표 6)',
    ' 오류의 88.4%가 경계 1픽셀 이내에 집중되고 상 면적비 오차는 0.3~1.3%p(Primary Si 0.31, Eutectic Si 0.74, Al₃Ni 1.15, Al 1.30%p)로 작아, 남은 개선 여지는 상을 통째로 놓치는 것보다 경계 정밀화에 있음')
ps[301] = sub(ps[301], '  4-1. 추가 분석', '  4-1. 추가 분석 및 인사이트')
ps[303] = ps[303] + boundary
ps[304] = sub(ps[304], '  4-2. 인사이트 및 발전 가능성', '  4-2. 향후 연구')
ps[305] = sub(ps[305], ' 오류의 88.4%가 경계 1픽셀 이내에 집중됨(상 면적비 오차: Primary Si 0.31, Eutectic Si 0.74, Al₃Ni 1.15, Al 1.30%p). 경계 학습을 강화하고, 상 면적비·형상 측정에 미치는 영향을 평가할 계획',
                   ' 경계 학습을 강화하고, 경계 오차가 상 면적비·형상 측정에 미치는 영향을 평가할 계획')
verify = sub(ps[310], ' 모델 선택과 보정값 조정에 사용하지 않은 별도 시편에서 경계 품질·측정 오차·검토 효율을 평가해 적용 가능성을 확인할 계획',
                      ' 위 항목은 모델 선택과 보정값 조정에 사용하지 않은 별도 시편에서 경계 품질·측정 오차·검토 효율을 평가해 적용 가능성을 확인할 계획')
ps[309], ps[310], ps[311] = '', ps[311], verify
out, last = [], 0
for i, m in enumerate(P):
    out.append(x[last:m.start()]); out.append(ps[i]); last = m.end()
out.append(x[last:]); x2 = ''.join(out)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist(): o.writestr(it, x2.encode() if it.filename == 'word/document.xml' else z.read(it.filename))
print('ok')
