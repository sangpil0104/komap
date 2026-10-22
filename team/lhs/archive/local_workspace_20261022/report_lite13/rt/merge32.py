"""Fold section 3-2 (originality) of the insight version into 3-1 R/E/F bullets, renumber 3-3/3-4."""
import re, sys, zipfile
src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src); x = z.read('word/document.xml').decode()
P = list(re.finditer(r'<w:p[ >].*?</w:p>', x, re.S))
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
txt = lambda p: ''.join(m.group(2) for m in T.finditer(p))
def sub(p, old, new):
    ms = [m for m in T.finditer(p) if m.group(2) == old]; assert len(ms) == 1, old[:40]
    m = ms[0]; return p[:m.start(2)] + new + p[m.end(2):]
ps = [m.group(0) for m in P]
assert txt(ps[181]).strip() == '3-2. 독창성 및 신기술 적용' and txt(ps[187]).strip() == '3-3. 모델링 방법' and txt(ps[262]).strip() == '3-4. 성능'
tpl, lab, body = ps[183], '선택적 보정의 효과:', ' E 단독 점수(79.59%)는 R(79.80%)보다 낮지만, E가 Al₃Ni·Al로 판단한 픽셀을 교체하면 80.75%(+0.95%p). 한 백본만 사용한 E는 효과가 작아 EfficientNet-B4 40%·HRNet 60%로 결합'
def bullet(l, b): return sub(sub(tpl, lab, l), body, b)
new_bullets = [
    bullet('R (상별 결합):', ' 단독 최고 모델이 상마다 다름(Al₃Ni·Eutectic Si는 ResNet34, Primary Si는 EfficientNet-B4 가로 4000, Al은 HRNet, 표 4), 모델 전체를 평균하지 않고 상별로 확률을 교체하며 비율은 결합 효과로 선택(예: Eutectic Si는 HRNet 드문 상 가중이 75%, 그림 3), 백본별 단순 평균(78.0~79.1%)보다 높은 79.80%'),
    bullet('E (부분 덮어쓰기):', ' 단독 점수(79.59%)는 R(79.80%)보다 낮지만 Al₃Ni·Al로 판단한 픽셀만 교체하면 Al₃Ni IoU가 76.58% → 78.23%로 올라 80.75%(+0.95%p), EfficientNet-B4 40%·HRNet 60%로 결합'),
    bullet('F (Al 재확인):', ' 80.75% → 80.78%(+0.03%p), 다른 시드 5개로 다시 학습해도 80.778~80.781%로 모두 상승'),
]
ps[103] = sub(ps[103], ' 같은 이미지를 기존 밝기(실제 밝기 차이 보존), 상대 밝기 Z(이미지 평균 대비 밝기, 노출 차이 제거), 밝기 순위 CDF(이미지 안에서 그보다 어둡거나 같은 픽셀의 비율 0~1, 노출이 달라도 순서 유지)로 표현',
                   ' 같은 이미지를 기존 밝기(실제 밝기 차이 보존), 상대 밝기 Z(이미지 평균 대비 밝기, 노출 차이 제거), 밝기 순위 CDF(이미지 안에서 그보다 어둡거나 같은 픽셀의 비율 0~1, 노출이 달라도 순서 유지)로 표현, Z로 Al₃Ni·Eutectic Si의 밝기 겹침 해소(표 3)')
ps[115] = sub(ps[115], ' 한 모델로 네 상을 모두 잘 맞히기 어려워, 상별 결합 앙상블(R)·전문가 앙상블(E)·최종 보정기(F)의 3단계로 결합해 최종 마스크 생성(그림 2)',
                   ' 한 모델로 네 상을 모두 잘 맞히기 어려워, 상별 결합 앙상블(R)·전문가 앙상블(E)·최종 보정기(F)의 3단계로 결합해 최종 마스크 생성(그림 2), 단독 점수가 아니라 상별 강점과 기존 오류 보완을 기준으로 모델을 결합한 것이 핵심')
ps[121] = ps[121] + ''.join(new_bullets)
ps[187] = sub(ps[187], '  3-3. 모델링 방법', '  3-2. 모델링 방법')
ps[262] = sub(ps[262], '  3-4. 성능', '  3-3. 성능')
for i in range(181, 187): ps[i] = ''
out, last = [], 0
for i, m in enumerate(P):
    out.append(x[last:m.start()]); out.append(ps[i]); last = m.end()
out.append(x[last:]); x2 = ''.join(out)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist(): o.writestr(it, x2.encode() if it.filename == 'word/document.xml' else z.read(it.filename))
print('ok')
