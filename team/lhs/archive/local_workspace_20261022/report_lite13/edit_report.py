"""Update the unpacked lite20 report (work/) to the lite13 model: text, tables, section 4, figures 2-4."""
import re, shutil
from pathlib import Path
from xml.sax.saxutils import escape

O = Path(__file__).resolve().parent
DOC = O / 'work/word/document.xml'
x = DOC.read_text()
PARA = re.compile(r'<w:p[ >].*?</w:p>', re.S)
TNODE = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')


def unesc(s):
    return s.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')


def rep(old, new, count=1):
    """Replace visible text `old` (may span runs) inside one paragraph; formatting of the first run is kept."""
    global x
    hits = 0
    out = []; last = 0
    for pm in PARA.finditer(x):
        p = pm.group()
        nodes = list(TNODE.finditer(p))
        texts = [unesc(n.group(2)) for n in nodes]
        joined = ''.join(texts)
        if old not in joined:
            continue
        hits += joined.count(old)
        start = joined.index(old); end = start + len(old)
        pos = 0; newtexts = []
        placed = False
        for t in texts:
            a, b = pos, pos + len(t); pos = b
            if b <= start or a >= end:
                newtexts.append(t); continue
            keep_pre = t[:max(0, start - a)]; keep_post = t[max(0, end - a):] if end < b else ''
            if not placed:
                newtexts.append(keep_pre + new + keep_post); placed = True
            else:
                newtexts.append(keep_post)
        q = []; pl = 0
        for n, t in zip(nodes, newtexts):
            q.append(p[pl:n.start()])
            open_tag = n.group(1)
            if 'xml:space' not in open_tag:
                open_tag = '<w:t xml:space="preserve">'
            q.append(open_tag + escape(t) + n.group(3)); pl = n.end()
        q.append(p[pl:])
        out.append(x[last:pm.start()]); out.append(''.join(q)); last = pm.end()
    assert hits == count, (old, hits)
    out.append(x[last:]); x = ''.join(out)


# ---------------------------------------------------------------- section 2
rep('(B4 8방향, HRNet 4방향)', '(E의 B4·R의 세 밝기 B4는 8방향, E의 가로 폭 정규화 HRNet은 4방향)')

# ---------------------------------------------------------------- section 3-1
rep('R(9개)과 E(6개)의 예측을 덮어쓰기로 합친 뒤 F(1개)로 보정(가중치 16개)',
    'R(7개)과 E(5개)의 예측을 덮어쓰기로 합친 뒤 F(1개)로 보정(가중치 13개)')
rep('상별로 골라 합친 R은 네 상 모두 단독 모델보다 높고, E로 Al₃Ni·Al을 교체하면 네 상 모두 추가 상승',
    '상별로 골라 합친 R은 Al을 뺀 세 상에서 단독 최고 모델보다 높고, E로 Al₃Ni·Al을 교체하면 네 상 모두 단독 최고 모델보다 높아짐')
rep('구성 요소 조합 중 검증 점수 80.6%로 가장 높은 9개 조합 선택, 검증용 이미지를 하나씩 빼고 다시 고르는 교차 검증으로 선택이 우연인지 확인',
    '검증 점수가 가장 높은 조합에서 시작해 점수를 떨어뜨리지 않는 범위에서 모델을 줄여 R 7개로 확정(④ 참고), 검증용 이미지를 하나씩 빼고 다시 고르는 교차 검증으로 선택이 우연인지 확인')
rep('E 단독 점수(79.58%)는 R보다 낮지만 Al₃Ni만 교체해도 +0.21%p, Al까지 교체하면 +0.24%p, 한 백본으로만 만든 E는 −0.08%p 또는 +0.11%p로 효과가 작아 두 백본을 평균',
    'E 단독 점수(79.62%)는 R(79.81%)보다 낮지만 R의 Al₃Ni·Al 픽셀만 E로 교체하면 80.68%(+0.88%p), 한 백본으로만 만든 E는 효과가 작아 B4와 HRNet을 반씩 결합')
rep('검증 점수 80.50% → 80.55%(+0.04%p), 다른 시드로 전체 재학습 시에도 +0.03%p로 같은 방향, 효과는 작지만 두 번의 학습에서 모두 상승해 유지',
    '검증 점수 80.68% → 80.72%(+0.04%p), 다른 시드 5개로 다시 학습해도 80.719~80.721%로 같은 효과, 효과는 작지만 모든 시드에서 상승해 유지')
rep('R 결합(11개)', 'R 결합(7개)')
rep('상별 결합·후처리·덮어쓰기 후 80.55%로, 단일 모델 대비 약 2%p 향상', '상별 결합·후처리·덮어쓰기 후 80.72%로, 단일 모델 대비 약 2.3%p 향상')

# ---------------------------------------------------------------- tables with repeated numbers: edit by row
def row_replace(row_key, cells_new):
    """Replace all text cells of the table row containing row_key (cells in order)."""
    global x
    i = x.index(row_key)
    a = max(x.rfind('<w:tr ', 0, i), x.rfind('<w:tr>', 0, i)); b = x.index('</w:tr>', i) + 7
    row = x[a:b]
    cells = re.findall(r'<w:tc>.*?</w:tc>', row, re.S)
    assert len(cells) == len(cells_new), (row_key, len(cells))
    new_row = row
    for c, txt in zip(cells, cells_new):
        nodes = list(TNODE.finditer(c))
        if not nodes:
            continue
        nc = c[:nodes[0].start()] + '<w:t xml:space="preserve">' + escape(txt) + '</w:t>' + c[nodes[0].end():]
        for n in nodes[1:]:
            nc = nc.replace(n.group(0), '<w:t></w:t>', 1)
        new_row = new_row.replace(c, nc, 1)
    x = x[:a] + new_row + x[b:]


# fixed: rows in the model comparison table
row_replace('R 결합(7개)', ['R 결합(7개)', '79.81%', '76.61%', '87.02%', '94.16%', '61.43%'])
row_replace('E 전문가 앙상블 단독', ['E 전문가 앙상블 단독', '79.62%', '75.85%', '86.58%', '94.50%', '61.56%'])
row_replace('최종(R + E + F)', ['최종(R + E + F)', '80.72%', '78.18%', '87.32%', '94.57%', '62.82%'])

# ---------------------------------------------------------------- section 3-2 training table
rep('R: ResNet34 U-Net++ + ASPP (3개)', 'R: ResNet34 U-Net++ + ASPP (2개)')
rep('R: EfficientNet-B4 U-Net 3종 (6개)', 'R: EfficientNet-B4 U-Net 2종 (4개)')
# HRNet: drop the basic-model row, keep one row for the rare-class weighted model
i = x.index('R: HRNet-W48 U-Net++ 드문 상 가중 (1개)')
a = max(x.rfind('<w:tr ', 0, i), x.rfind('<w:tr>', 0, i)); b = x.index('</w:tr>', i) + 7
x = x[:a] + x[b:]
rep('R: HRNet-W48 U-Net++ 기본 (1개)', 'R: HRNet-W48 U-Net++ 드문 상 가중 (1개)')
rep('70 epoch, batch 8', '70 epoch 후 드문 상 가중 12 epoch 추가, batch 8')
i = x.index('원본 256×256 영역을 512×512로 확대'); a = x.rfind('<w:vMerge w:val="restart"/>', 0, i)
assert i - a < 400
x = x[:a] + x[a + len('<w:vMerge w:val="restart"/>'):]
rep('E: EfficientNet-B4 U-Net (4개)', 'E: EfficientNet-B4 U-Net (2개)')
rep('E: HRNet-W48 U-Net++ (4개)', 'E: HRNet-W48 U-Net++ (3개)')
rep('가로 폭 정규화 후 2배 확대, 512×512 영역', '가로 폭 정규화 후 2배 확대(1개는 세 밝기 표현 입력), 512×512 영역')
rep('70 epoch, 시드 2개 × fold 4개 중 4개 사용', '70 epoch(fold 4 모델, 학습용 70장 전체 모델), 세 밝기 입력 모델은 25 epoch 추가 학습')

# ---------------------------------------------------------------- same-backbone bullets
rep('ResNet34 고해상도 4개: 768×768 크기로 잘라 추가 학습한 고해상도 기본 모델, Al₃Ni·Eutectic Si 사이의 확률 차이를 벌리는 손실을 더한 고해상도 모델, 이미지를 줄이지 않고 원래 크기 그대로 처리하는 원본 해상도 모델로 구성, 고해상도는 기본 모델을 쓰되 검증 데이터로 정한 좁은 밝기 구간(이미지 평균 밝기 159~161)에서만 손실 모델로 교체',
    'ResNet34 2개: Al₃Ni·Eutectic Si 사이의 확률 차이를 벌리는 손실을 더해 768×768 크기로 추가 학습한 고해상도 모델(모든 이미지의 기본 확률)과, 이미지를 줄이지 않고 원래 크기 그대로 처리하는 원본 해상도 모델(Eutectic Si 결합)로 구성')
rep('B4 6개: 입력에 따라 흑백 밝기 1개, 세 밝기 표현 2개, 가로 4000픽셀 맞춤 3개로 구분, 각 모델은 4묶음 중 하나를 뺀 약 52장으로 학습했고, 같은 입력끼리는 빼 둔 묶음만 다름(흑백 fold 3, 세 밝기 fold 1·2, 가로 4000픽셀 fold 1·3·4)',
    'B4 4개: 세 밝기 표현 1개(fold 2, 새 시드로 재학습)와 가로 4000픽셀 맞춤 3개(fold 1·3·4)로 구성, 각 모델은 4묶음 중 하나를 뺀 약 52장으로 학습, Al 결합에 쓰던 흑백 B4는 세 밝기 B4의 확률로 대신해 제거(④ 참고)')
rep('HRNet 2개: 기본 모델과, 여기에 Eutectic Si·Primary Si 오차 가중치를 줘 12 epoch 추가 학습한 모델, 두 번째 모델은 Al 확률에만 25% 반영',
    'HRNet 1개: 기본 HRNet에 Eutectic Si·Primary Si 오차 가중치를 줘 12 epoch 추가 학습한 드문 상 가중 모델, 기본 HRNet이 맡던 자리까지 같은 확률로 겸함(그림 3)')
rep('E 8개: B4 4개는 R에도 쓰인 가로 4000픽셀 B4를 fold별로 추가 학습한 모델, HRNet 4개는 가로 폭을 맞춰 학습해 R의 HRNet과 다른 크기로 이미지를 봄(시드 2개 × 4-fold 중 시드 1의 fold 2·4, 시드 2의 fold 1·4 사용)',
    'E 5개: B4 2개는 R에도 쓰인 가로 4000픽셀 B4의 fold 1·3을 추가 학습한 모델, HRNet 3개는 가로 폭을 맞춰 학습한 fold 4 모델, 같은 방식으로 학습용 70장 전체를 학습한 모델, 기본 HRNet을 세 밝기 표현 입력으로 추가 학습한 모델로 구성해 R의 HRNet과 다른 크기·입력으로 이미지를 봄')
rep('E는 B4 평균 50%, HRNet 두 시드의 평균 각 25%로 결합',
    'E는 B4 2개 평균 50%, 세 밝기 입력 HRNet·fold 4 HRNet 평균 25%, 70장 전체 학습 HRNet 25%로 결합')

# ---------------------------------------------------------------- section 3-3
rep('20장 평균 80.546432%', '20장 평균 80.720984%')
for key, cells in [('89.6%', ['Al₃Ni', '78.18%', '89.5%', '90.5%', 'Al']), ('94.9%', ['Primary Si', '87.32%', '95.1%', '95.4%', 'Eutectic Si']),
                   ('98.0%', ['Al', '94.57%', '97.9%', '96.9%', 'Al₃Ni']), ('75.3%', ['Eutectic Si', '62.82%', '76.9%', '80.7%', 'Al₃Ni'])]:
    i = x.index('>' + key + '<'); a = max(x.rfind('<w:tr ', 0, i), x.rfind('<w:tr>', 0, i)); row = x[a:x.index('</w:tr>', i)]
    k0 = re.search(r'<w:t[^>]*>([^<]+)</w:t>', row).group(1)
    row_replace_key = key
    # replace by locating the row through its unique recall cell
    tmp = x.index('>' + key + '<')
    a2 = max(x.rfind('<w:tr ', 0, tmp), x.rfind('<w:tr>', 0, tmp)); b2 = x.index('</w:tr>', tmp) + 7
    row = x[a2:b2]; cs = re.findall(r'<w:tc>.*?</w:tc>', row, re.S); assert len(cs) == 5
    nr = row
    for c, t in zip(cs, cells):
        nodes = list(TNODE.finditer(c))
        nc = c[:nodes[0].start()] + '<w:t xml:space="preserve">' + escape(t) + '</w:t>' + c[nodes[0].end():]
        for n in nodes[1:]:
            nc = nc.replace(n.group(0), '<w:t></w:t>', 1)
        nr = nr.replace(c, nc, 1)
    x = x[:a2] + nr + x[b2:]
i = x.index('>전체 평균<'); a2 = max(x.rfind('<w:tr ', 0, i), x.rfind('<w:tr>', 0, i)); b2 = x.index('</w:tr>', i) + 7
x = x[:a2] + x[a2:b2].replace('80.55%', '80.72%') + x[b2:]
rep('최저인 A71(54.45%)은 Primary Si가 원래 없어 규칙상 0점이 포함됨(나머지 세 상 평균 72.59%)',
    '최저인 A71(53.68%)은 Primary Si가 원래 없어 규칙상 0점이 포함됨(나머지 세 상 평균 71.58%)')
rep('틀린 픽셀의 85.6%가 정답 경계 1픽셀 이내', '틀린 픽셀의 88.7%가 정답 경계(이웃 픽셀과 상이 바뀌는 곳) 1픽셀 이내')
rep('정답 Al₃Ni를 Al로 예측한 픽셀 약 19만, Eutectic Si로 예측한 픽셀 약 14만, 반대로 정답 Eutectic Si를 Al₃Ni로 예측한 픽셀 약 15만',
    '정답 Al₃Ni를 Al로 예측한 픽셀 약 18만, Eutectic Si로 예측한 픽셀 약 15만, 반대로 정답 Eutectic Si를 Al₃Ni로 예측한 픽셀 약 14만')
rep('Al은 정답의 98.0%를 찾지만 Eutectic Si는 75.3%만 찾고 예측의 81.9%만 맞아',
    'Al은 정답의 97.9%를 찾지만 Eutectic Si는 76.9%만 찾고 예측의 80.7%만 맞아')
rep('(A153·A74는 Eutectic Si IoU 28.7%·40.3%, A11_step1·A67은 Al₃Ni IoU 56.6%·58.4%)',
    '(A153·A74는 Eutectic Si IoU 31.3%·40.4%, A67·A11_step2는 Al₃Ni IoU 59.4%·64.9%)')

# ---------------------------------------------------------------- section 4 (replaces the placeholder paragraph)
def ppr(before=None):
    sp = f'<w:spacing w:before="{before}" w:line="300" w:lineRule="auto"/>' if before else '<w:spacing w:line="300" w:lineRule="auto"/>'
    return f'<w:pPr><w:pStyle w:val="a6"/><w:keepLines/>{sp}<w:ind w:left="410" w:right="176" w:hanging="410"/><w:jc w:val="left"/></w:pPr>'


def run(t, bold=False):
    b = '<w:rPr><w:b/><w:bCs/></w:rPr>' if bold else ''
    return f'<w:r>{b}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'


def head(t):
    return f'<w:p>{ppr().replace("<w:keepLines/>", "<w:keepNext/><w:keepLines/>")}{run("  " + t, True)}</w:p>'


def sq(label, t):
    return f'<w:p>{ppr()}{run("  □ ")}{run(label + ":", True)}{run(" " + t)}</w:p>'


def dash(label, t):
    lab = run(label + ":", True) + run(" " + t) if label else run(t)
    return f'<w:p>{ppr(80)}{run("   - ")}{lab}</w:p>'


SEC4 = [
    head('4-1. 경량화: 가중치 20개 → 13개'),
    sq('방법', '최종 모델에서 구성원을 하나씩 빼거나 다른 구성원의 확률을 같이 쓰도록 바꾸고, 검증 점수가 떨어지지 않고 어떤 검증 이미지도 0.1%p 넘게 나빠지지 않을 때만 채택'),
    dash('결과', 'R 11개 → 7개, E 8개 → 5개로 줄여 가중치 20개(80.55%) → 13개(80.72%), 같은 확률을 두 자리에 쓰는 공유 2곳(흑백 B4 → 세 밝기 B4, 기본 HRNet → 드문 상 가중 HRNet)은 추가 학습 없이 점수 유지'),
    dash('E 재구성', 'HRNet 두 시드 × fold 2개(4개)를 fold 4 모델, 70장 전체 학습 모델, 세 밝기 입력 모델의 3개로 바꾸고 B4를 fold 1·3의 2개로 줄임'),
    dash('줄이지 못한 부분', '가로 4000픽셀 B4 3개는 fold 2개로 줄이면 −0.04~−0.06%p, 1개면 −0.3%p 이상이며, 남긴 fold에 TTA·확률 보정, 가중치 평균, 인코더 공유, 다른 모델로 대체, 70장 재학습(EMA·SWA) 모두 A71·A186에서 1~6%p 손실이 커 기각'),
    dash(None, 'E의 B4를 1개로 줄이면 평균은 같지만(+0.001%p) 20장 중 14장이 나빠지고 시험용 예측의 0.42%가 바뀌어 보류'),
    head('4-2. 점수의 신뢰도'),
    sq('선택 편향', '검증용 20장을 구성원 선택에도 썼으므로 80.72%에는 선택에 따른 낙관이 포함됨'),
    dash('시드 편차', '구성원을 다른 시드로 학습한 모델로 바꾸면 점수의 표준편차는 0.001~0.04%p이고, 현재 쓰는 세 밝기 B4·70장 전체 학습 HRNet은 시드 평균보다 0.05%p·0.03%p 높음'),
    dash('결합 비율', '20장 전체로 비율을 다시 고르면 +0.06%p이지만, 시편을 반으로 나눠 한쪽에서 고른 비율을 다른 쪽에 쓰면 −1.12%p~+0.03%p로 일정하지 않아 비율은 바꾸지 않음'),
    dash('모델 추가', '기본 HRNet 계열을 E에 10% 더하면 +0.03%p로 여러 변형에서 일관되지만, 가중치 1개 증가 대비 효과가 작아 채택하지 않음'),
    head('4-3. 향후 연구'),
    sq('경계 정밀화', '틀린 픽셀의 88.7%가 경계 1픽셀 이내이므로, 더 높은 해상도 입력과 경계 손실 강화가 남은 개선 여지'),
    sq('추가 경량화', '가로 4000픽셀 B4 3개의 평균 확률을 그대로 따라 하도록 한 모델을 학습(증류)하면 가중치를 2개 더 줄일 수 있으나, 기존 시도는 −0.25%p로 학습 방식 개선이 필요'),
    sq('평가 신뢰도', '검증 이미지가 20장(시편 19개)뿐이라 0.05%p 이하 차이는 구분하기 어려우므로, 시편 단위 교차 검증을 늘려 선택 편향을 줄일 필요'),
]
i = x.index('※ 최종 모델 확정 후 작성 예정')
a = max(x.rfind('<w:p ', 0, i), x.rfind('<w:p>', 0, i)); b = x.index('</w:p>', i) + 6
x = x[:a] + ''.join(SEC4) + x[b:]

# ---------------------------------------------------------------- 2-1 / 2-2 condensed under the required headings
def block_replace(first_key, last_key, paras):
    global x
    i = x.index(first_key); a = max(x.rfind('<w:p ', 0, i), x.rfind('<w:p>', 0, i))
    j = x.index(last_key, i); b = x.index('</w:p>', j) + 6
    old = [''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))[:40] for p in re.findall(r'<w:p[ >].*?</w:p>', x[a:b], re.S)]
    assert '<w:tbl>' not in x[a:b] and len(old) <= 8, old
    print('replacing:', old)
    x = x[:a] + ''.join(paras) + x[b:]


block_replace('세 가지 밝기 표현:', '입력 정규화:', [
    sq('세 가지 밝기 표현', '같은 이미지를 기존 밝기(실제 밝기 차이 보존), 상대 밝기 Z(이미지 평균 대비 밝기, 노출 차이 제거), 밝기 순위 CDF(이미지 안에서 그보다 어둡거나 같은 픽셀의 비율 0~1, 노출이 달라도 순서 유지)로 표현'),
    dash('입력 정규화', 'B4와 E의 세 밝기 HRNet은 세 표현을 3채널로 쌓아 입력(B4는 이미지 전체 평균·표준편차로 정규화), F는 Z를 ±3에서 잘라 0~1로 조정(이미지 평균 = 0.5)'),
    sq('크기 맞춤', '작은 조직을 잘 보도록 2~3배 확대하거나 가로 4000픽셀로 조정, 정답 마스크는 상 표시가 섞이지 않도록 최근접 보간(가장 가까운 픽셀 값 사용)'),
    dash('가로 폭 정규화(E)', '배율 차이를 줄이고자 B4는 4000픽셀, HRNet은 학습 fold에서 가장 흔한 폭(1230·1278픽셀)으로 맞춘 뒤 2배 확대, R과 다른 크기로 보아 두 앙상블의 오류 중복 감소'),
    sq('정답 변환', '색으로 칠한 정답 마스크를 상 번호(Al₃Ni 0, Primary Si 1, Al 2, Eutectic Si 3)로 변환'),
])
block_replace('데이터 증강:', '예측 시 증강(TTA)', [
    sq('학습용 4-fold 분할', '학습용 70장을 파일명 앞 번호(예: A71)가 같은 이미지끼리 묶어 4묶음(18·18·17·17장)으로 고정, 한 묶음씩 빼고 학습해 fold 모델 4개 생성'),
    sq('학습 영역 샘플링', '잘라내는 영역의 절반을 Al₃Ni·Eutectic Si 주변에서 선택, B4는 확대 이미지에서 1536×1536 영역(3배 확대 시 원본 약 512×512)을 자르고 가로 4000픽셀 B4는 이미지를 같은 빈도로, 나머지 B4는 면적에 비례해 선택'),
    sq('데이터 증강', '회전·뒤집기·확대·밝기 변화를 이미지와 정답 마스크에 같은 변환으로 적용해 다양한 촬영 조건 학습'),
    sq('예측 시 처리', '큰 이미지는 겹치게 잘라 예측해 이어 붙임(ResNet34 768·1024 영역을 576픽셀 간격, HRNet 512 영역을 170픽셀 겹침, B4는 전체 이미지), 회전·뒤집기 예측 평균(TTA: ResNet34 고해상도와 E의 B4·R의 세 밝기 B4 8방향, E의 가로 폭 정규화 HRNet 4방향) 후 원래 크기로 복원'),
])

# ---------------------------------------------------------------- numbered table captions; text refers to them by number
TABLES = [('① 밝기·배율 차이', '표 1. 점수를 좌우하는 핵심 과제와 단계별 대응'),
          ('학습용 픽셀 비율', '표 2. 상별 픽셀 비율과 그 상이 들어 있는 이미지 수'),
          ('상대 밝기 Z', '표 3. 상별 밝기 범위(기존 밝기와 이미지 내 상대 밝기 Z)'),
          ('모델(검증용 단독)', '표 4. 모델별 검증용 단독 점수와 결합 단계별 점수(IoU)'),
          ('입력과 크기', '표 5. 구성 모델별 입력과 학습량'),
          ('주로 헷갈린 상', '표 6. 최종 모델의 상별 검증 성능')]
for key, cap in TABLES:
    i = x.index(key); a = x.rfind('<w:tbl>', 0, i)
    p = ('<w:p><w:pPr><w:pStyle w:val="a6"/><w:keepNext/><w:keepLines/><w:spacing w:before="60" w:after="40" w:line="250" w:lineRule="auto"/>'
         '<w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr><w:t xml:space="preserve">' + escape(cap) + '</w:t></w:r></w:p>')
    x = x[:a] + p + x[a:]
rep('전처리, 모델링에서 아래와 같이 대응', '전처리, 모델링에서 표 1과 같이 대응')
rep('학습용·검증용별로 계산(아래 표)', '학습용·검증용별로 계산(표 2)')
rep('두 상이 거의 분리됨(아래 표)', '두 상이 거의 분리됨(표 3)')
rep('상마다 가장 잘 맞히는 모델이 다름(아래 표)', '상마다 가장 잘 맞히는 모델이 다름(표 4)')
rep('(쓰는 fold는 아래 참고)', '(쓰는 fold는 표 5 참고)')
rep('놓침·과잉 예측 모두 많음(위 표)', '놓침·과잉 예측 모두 많음(표 6)')
i = x.index(' ④ 추가 분석 및 향후 연구'); a = x.rfind('<w:pPr>', 0, i)
x = x[:a] + '<w:pPr><w:pageBreakBefore/>' + x[a + len('<w:pPr>'):]
DOC.write_text(x)
for src, dst in [('figure2_lite13.png', 'image2.png'), ('figure3_lite13.png', 'image3.png'), ('figure4_lite13.png', 'image4.png')]:
    shutil.copyfile(O / src, O / 'work/word/media' / dst)
print('edited')
