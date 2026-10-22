"""Apply review fixes to the teammate's insight version (whole-run text replacement, formatting kept)."""
import re, sys, zipfile
src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src); x = z.read('word/document.xml').decode()
P = list(re.finditer(r'<w:p[ >].*?</w:p>', x, re.S))
T = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
E = {  # paragraph index -> [(old run text, new run text)]
 16: [(' 네 상의 분할을 자동화하고, 상 면적비 산출과 조직 경계 검토에 활용',
       ' 네 상의 분할을 자동화하고, 상 면적비 산출과 조직 경계 검토에 활용(검증용 상 면적비 오차 평균 0.3~1.3%p)')],
 17: [(' 모델 간 예측 불일치 영역을 표시해 전문가 검토와 추가 라벨링의 우선순위 선정으로 확장',
       ' 모델 간 예측 불일치 영역을 전문가 검토·추가 라벨링 우선순위 선정에 활용하는 방향으로 확장 가능(4-2에서 검증 계획)')],
 120: [('%까지 상승', '%(표 4)')],
 121: [(' 단독 최고인 EfficientNet-B4(78.83%)로 시작하면 최종 점수가 0.06~0.09%p 낮아 ResNet34 고해상도 모델을 채택',
        ' 단독 최고인 EfficientNet-B4(78.83%)로 시작하면 최종 점수가 0.02~0.04%p 낮아 ResNet34 고해상도 모델을 채택')],
 182: [(' 상별 강점과 결합 효과를 기준으로 R의 모델 비중을 정하고, E·F는 정해진 상에 선택적으로 개입하도록 설계. 단독 점수와 기존 오류를 보완하는 능력을 함께 고려한 것이 차별점',
        ' 상별 강점과 결합 효과로 R의 모델 비중을 정하고 E·F는 정해진 상에만 개입하도록 설계해, 백본별 단순 평균(78.0~79.1%) 대비 80.78% 달성. 단독 점수와 기존 오류 보완 능력을 함께 고려한 것이 차별점')],
 185: [('촬영 조건에 대응한 기술 적용:', '촬영 조건에 대응한 입력 설계:'),
       (' 원본·상대 밝기 Z·밝기 순위 CDF를 입력에 활용하고, 예측 시 적응 기술 TENT로 정답 없이 BatchNorm을 이미지별로 조정. 밝기 표현과 예측 단계의 적응을 함께 적용하도록 구성',
        ' 원본 밝기·상대 밝기 Z·밝기 순위 CDF를 함께 입력해 이미지별 노출 차이에 강하게 구성')],
 241: [(' Eutectic Si·Primary Si 오차 가중치로 추가 학습. 단독 점수는 기본 HRNet보다 낮지만(표 4), 결합에서는 Al·Eutectic Si를 주로 맡고 기본 HRNet의 비중까지 겸함(그림 3)',
        ' 기본 HRNet(U-Net++ 디코더·경계 보조 출력, 70 epoch, 최종 구성에는 미포함)에 Eutectic Si·Primary Si 오차 가중치를 줘 12 epoch 추가 학습. 단독 점수는 기본 HRNet보다 낮지만(표 4), 결합에서는 Al·Eutectic Si를 주로 맡고 기본 HRNet의 비중까지 겸함(그림 3)')],
 308: [(' 오류의 88.4%가 경계 1픽셀 이내에 집중됨. 경계 학습을 강화하고, 상 면적비·형상 측정에 미치는 영향을 평가할 계획',
        ' 오류의 88.4%가 경계 1픽셀 이내에 집중됨(상 면적비 오차: Primary Si 0.31, Eutectic Si 0.74, Al₃Ni 1.15, Al 1.30%p). 경계 학습을 강화하고, 상 면적비·형상 측정에 미치는 영향을 평가할 계획')],
 314: [(' 가로 4000픽셀 EfficientNet-B4 3개를 1개에 증류해 가중치 2개 축소를 목표로 함. 기존 시도는 실패하여 학습 방식을 개선하고 성능·추론 시간을 함께 평가할 계획',
        ' 가로 4000픽셀 EfficientNet-B4 3개를 1개에 증류해 가중치 2개 축소를 목표로 함. 기존 시도는 실패하여 학습 방식을 개선하고 성능·추론 시간을 함께 평가할 계획. 현재도 정밀(13개, 80.78%)·빠른(EfficientNet-B4 1개, 78.38%) 모드 선택 가능')],
}
new = {}
for i, reps in E.items():
    p = P[i].group(0)
    for old, rep in reps:
        ms = [m for m in T.finditer(p) if m.group(2) == old]
        assert len(ms) == 1, (i, old[:30], len(ms))
        m = ms[0]; p = p[:m.start(2)] + rep + p[m.end(2):]
    new[i] = p
out, last = [], 0
for i, m in enumerate(P):
    out.append(x[last:m.start()]); out.append(new.get(i, m.group(0))); last = m.end()
out.append(x[last:]); x2 = ''.join(out)
ds = [m for m in re.finditer(r'<w:drawing>.*?</w:drawing>', x2, re.S) if 'r:embed="rId11"' in m.group(0)]
assert len(ds) == 1; d = ds[0].group(0); e = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', d)
cx, cy = int(e.group(1)), int(e.group(2)); assert d.count(f'cx="{cx}" cy="{cy}"') == 2  # figure 4 to 82%
x2 = x2[:ds[0].start()] + d.replace(f'cx="{cx}" cy="{cy}"', f'cx="{round(cx*.82)}" cy="{round(cy*.82)}"') + x2[ds[0].end():]
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for it in z.infolist(): o.writestr(it, x2.encode() if it.filename == 'word/document.xml' else z.read(it.filename))
print('ok')
