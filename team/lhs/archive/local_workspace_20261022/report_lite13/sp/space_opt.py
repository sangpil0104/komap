"""Space-only layout fixes on the user's report (no text change). usage: space_opt.py IN_DIR"""
import re, sys
D = sys.argv[1]; p = f'{D}/word/document.xml'
x = open(p).read(); head = x[:x.index('<w:body>')]; body = x[x.index('<w:body>'):]
parts = re.split(r'(<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>)', body, flags=re.S)
blocks = [i for i, s in enumerate(parts) if s.startswith(('<w:p', '<w:tbl>')) and not s.startswith('<w:pPr')]
txt = lambda s: ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', s))
def B(k): return parts[blocks[k]]
def setB(k, v): parts[blocks[k]] = v
def find(s): return [k for k in range(len(blocks)) if txt(B(k)).strip().startswith(s)][0]
log = []
# 1) empty spacer paragraphs right before '③ 모델링'
k3 = find('③ 모델링')
for k in range(k3 - 1, k3 - 6, -1):
    if B(k).startswith('<w:p') and not txt(B(k)).strip() and '<w:drawing' not in B(k):
        setB(k, ''); log.append(f'removed empty para before ③ ({k})')
    else:
        break
# 2) pageBreakBefore on '④ 추가 분석'
k4 = find('④ 추가 분석')
setB(k4, B(k4).replace('<w:pageBreakBefore/>', '')); log.append('removed pageBreakBefore on ④')
# 3) break the 3-3 keep-with-next chain: keep heading->first line, caption->table, '시험용 이미지 결과'->figure
for s in ('□ 검증 방법', '- 이미지별 점수', '□ 오류 위치', '- 자주 헷갈린 상', '- 상별 정확도', '- 점수가 낮은 이미지', '□ 모델 규모', '□ 추론 시간', '□ 단일 모델 대비'):
    try:
        k = find(s)
    except IndexError:
        continue
    setB(k, B(k).replace('<w:keepNext/>', '', 1)); log.append(f'keepNext off: {s}')
# table 6: no keepNext on its last row so it does not chain to the following text
k6 = [k for k in range(len(blocks)) if B(k).startswith('<w:tbl>') and '표 6.' in txt(B(k))][0]
t = B(k6); rows = re.findall(r'<w:tr[ >].*?</w:tr>', t, re.S); last = rows[-1]
t = t.replace(last, last.replace('<w:keepNext/>', '')); setB(k6, t); log.append('table 6 last row keepNext off')
open(p, 'w').write(head + ''.join(parts)); print('\n'.join(log))
# 4) table 5 sits inside a one-cell wrapper table; Word will not split a nested table across pages, so the whole
#    table jumps to the next page. Unwrap it: caption paragraph + inner table become top-level blocks.
x = open(p).read(); i = x.index('표 5. 구성'); a = x.rfind('<w:tbl>', 0, i)
depth, j = 0, a
for m in re.finditer(r'<w:tbl>|</w:tbl>', x[a:]):
    depth += 1 if m.group() == '<w:tbl>' else -1
    if depth == 0:
        b = a + m.end(); break
outer = x[a:b]
c0 = outer.index('</w:tcPr>') + len('</w:tcPr>'); c1 = outer.rindex('</w:tc>')
inner = outer[c0:c1]
tail = re.search(r'(<w:p[ >](?:(?!<w:p[ >]).)*?</w:p>)$', inner, re.S)
if tail and not re.findall(r'<w:t[^>]*>([^<]+)</w:t>', tail.group(1)):
    inner = inner[:tail.start()]
x = x[:a] + inner + x[b:]; open(p, 'w').write(x); print('table 5 unwrapped from its wrapper table')
# 5) body line spacing 300 -> 288 (about 4% tighter)
x = open(p).read(); n = x.count('w:line="300" w:lineRule="auto"')
x = x.replace('w:line="300" w:lineRule="auto"', 'w:line="288" w:lineRule="auto"'); open(p, 'w').write(x); print(f'line spacing 300->288 on {n} paragraphs')
