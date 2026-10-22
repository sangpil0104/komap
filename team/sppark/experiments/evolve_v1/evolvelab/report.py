"""Results summary (CSV + short Markdown leaderboard). A deliberately simplified counterpart to
experiments/hundred_v1/hundredlab/report.py -- enough to satisfy the suite() driver's contract and
give a readable leaderboard, without replicating that file's full chart/report machinery."""
import csv

from evolvelab.catalog import catalog
from evolvelab.workspace import output_path, read_json


def report(failures=None, smoke=False, publish=True):
    failures = failures or {}
    entries = catalog()
    rows = []
    for name, item in entries.items():
        summary_path = output_path('smoke_evaluations' if smoke else 'evaluations', name, 'd4', 'metrics_summary.json')
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        rows.append(dict(experiment=name, parent=item['parent'], group=item['group'],
                          description=item['description'], image_mIoU=summary['image_mIoU'],
                          global_mIoU=summary['global_mIoU']))
    rows.sort(key=lambda r: -r['image_mIoU'])
    all_done = len(rows) + len(failures) >= len(entries)
    if publish:
        out_dir = output_path('reports')
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / ('smoke_results.csv' if smoke else 'results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['experiment', 'parent', 'group', 'description', 'image_mIoU', 'global_mIoU'])
            writer.writeheader()
            writer.writerows(rows)
        md_path = out_dir / ('smoke_results.md' if smoke else 'results.md')
        lines = ['# evolve_v1 results', '', f'완료 {len(rows)}/{len(entries)}, 실패 {len(failures)}', '',
                 '| 순위 | 실험 | 부모 | 그룹 | 이미지 mIoU | Global mIoU |', '|---:|---|---|---|---:|---:|']
        for i, r in enumerate(rows, 1):
            lines.append(f"| {i} | {r['experiment']} | {r['parent']} | {r['group']} | {r['image_mIoU']:.4f} | {r['global_mIoU']:.4f} |")
        if failures:
            lines += ['', '## 실패', ''] + [f"- {name}: {reason}" for name, reason in failures.items()]
        md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(completed=len(rows), total=len(entries), failures=len(failures)), all_done
