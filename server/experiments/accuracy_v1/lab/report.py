from __future__ import annotations

import html
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from alphase.utils import write_json
from lab.workspace import BASELINE, output_path, read_json


def compare_results():
    baseline_file = BASELINE/"run/evaluation/per_image_metrics.csv"
    baseline = pd.read_csv(baseline_file).set_index("image_id")
    candidates = [("baseline / plain", baseline_file),
                  ("baseline / D4 (previous review)", BASELINE/"run/evaluation_d4_review/per_image_metrics.csv")]
    for branch in ("evaluations", "ensembles"):
        for path in sorted(output_path(branch).glob("*/official/valid/**/per_image_metrics.csv")):
            marker = path.parent/"provenance.json"
            if not marker.exists():
                continue
            data = read_json(marker)
            name = data.get("experiment", data.get("name"))
            if name == "baseline":
                continue  # The immutable baseline rows are already present above.
            inference = data["member"]["inference"] if branch == "evaluations" else data["members"][0]["inference"]
            suffix = "D4" if inference["d4"] else "plain"
            candidates.append((f"{name} / {suffix}", path))
    # Ensemble results live directly under .../valid, which ** also includes.
    result, paired = [], []
    for name, path in candidates:
        if not path.is_file():
            continue
        frame = pd.read_csv(path).set_index("image_id")
        if set(frame.index) != set(baseline.index) or len(frame) != len(baseline):
            raise ValueError(f"Comparison requires exactly the same 20 validation images: {path}")
        frame = frame.loc[baseline.index]
        delta = frame.mIoU_exclude-baseline.mIoU_exclude
        row = dict(experiment=name, images=len(frame), miou=float(frame.mIoU_exclude.mean()),
                   delta_pp=float(delta.mean()), improved_images=int((delta>1e-8).sum()),
                   worsened_images=int((delta< -1e-8).sum()), source=str(path))
        row.update({f"iou_{c}": float(frame[f"IoU_{c}"].mean()) for c in range(4)})
        result.append(row)
        paired.extend(dict(experiment=name, image_id=image_id, baseline_miou=float(baseline.loc[image_id, "mIoU_exclude"]),
                           candidate_miou=float(frame.loc[image_id, "mIoU_exclude"]), delta_pp=float(delta.loc[image_id])) for image_id in frame.index)
    frame = pd.DataFrame(result).sort_values("miou", ascending=False)
    out = output_path("reports")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/"comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(out/"per_image_comparison.csv", index=False)
    write_json(out/"comparison.json", dict(baseline_miou=float(baseline.mIoU_exclude.mean()), candidates=frame.to_dict("records"),
                                          note="Development validation scores on the same 20 images; not nested CV or independent final test scores."))
    lines = []
    for row in frame.to_dict("records"):
        path = Path(row["source"])
        if path.is_relative_to(BASELINE):
            target = out/("reference_baseline_d4.csv" if "D4" in row["experiment"] else "reference_baseline_plain.csv")
            shutil.copyfile(path, target)
            path = target
        href = html.escape(os.path.relpath(path, out), quote=True)
        name = html.escape(row["experiment"])
        cells = [f'<a href="{href}">{name}</a>', f'{row["miou"]:.3f}', f'{row["delta_pp"]:+.3f}',
                 f'{row["improved_images"]}/{row["images"]}']+[f'{row[f"iou_{i}"]:.2f}' for i in range(4)]
        lines.append('<tr>'+''.join(f'<td>{cell}</td>' for cell in cells)+'</tr>')
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlPhase 실험 비교</title><style>body{font:16px system-ui,sans-serif;background:#101820;color:#e6eef4;margin:40px auto;max-width:1400px;padding:0 20px}p{color:#b0c2d0;line-height:1.7}a{color:#79d9bb}table{border-collapse:collapse;width:100%;background:#192530}td,th{padding:15px;text-align:right;border-bottom:1px solid #32414e}td:first-child,th:first-child{text-align:left}th{cursor:pointer}input{background:#192530;color:white;border:1px solid #657785;padding:10px;width:300px;max-width:85%}.scroll{overflow:auto}small{color:#b0c2d0}</style>
<h1>정확도 개선 실험 비교</h1><p>기준 모델과 동일한 검증 20장의 이미지별 평균 mIoU입니다. 원본 모델과 결과는 보존되어 있습니다.<br>열 제목을 클릭하면 정렬됩니다. 아래 점수는 개발 검증 결과이며, 최종 선택은 group CV로 다시 확인하세요.</p>
<input id="filter" placeholder="실험 이름 검색"><p><a href="comparison.csv">전체 CSV</a> · <a href="per_image_comparison.csv">이미지별 차이 CSV</a></p><div class="scroll"><table><thead><tr><th>실험</th><th>mIoU (%)</th><th>기준 대비 (%p)</th><th>개선 이미지</th><th>초정 Si</th><th>Al₃Ni</th><th>공정 Si</th><th>Al</th></tr></thead><tbody>'''+''.join(lines)+'''</tbody></table></div><p>표에 아직 없는 실험은 학습/평가 완료 후 compare 명령을 다시 실행하면 표시됩니다. 기존 검증 기록은 baseline/에 복사된 자료에서 읽습니다.</p>
<script>document.querySelector('#filter').oninput=e=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()));document.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{const body=document.querySelector('tbody');const rows=[...body.rows];const direction=th.dataset.direction==='up'?-1:1;th.dataset.direction=direction===1?'up':'down';rows.sort((a,b)=>{const x=a.cells[i].textContent,y=b.cells[i].textContent;return direction*(i&&i!==3?parseFloat(y)-parseFloat(x):x.localeCompare(y))});rows.forEach(r=>body.append(r))});</script></html>'''
    (out/"comparison.html").write_text(document)
    print(frame[["experiment", "miou", "delta_pp", "improved_images"]].to_string(index=False))
    print(f"Report: {out/'comparison.html'}")
    return frame
