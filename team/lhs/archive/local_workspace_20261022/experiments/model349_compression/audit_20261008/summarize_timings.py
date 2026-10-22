"""Summarize component walltime and recorded shared-server conditions."""
import json, statistics
from datetime import datetime
from pathlib import Path
ROOT=Path('/workspace/model349_compress/audit_20261008')
def load(p):return json.loads(p.read_text())
result={'units':'seconds (component walltime, not CUDA kernel duration)','groups':{}}
for tag in ['batched','hramp']:
 p=ROOT/f'{tag}_summary.json'
 if not p.exists():continue
 summary=load(p); groups={}
 for kind in ['references','candidates']:
  rows=[]
  for name in summary.get(kind,[]):
   folder=ROOT/'runs'/name
   if not (folder/'completion.json').exists():continue
   completion=load(folder/'completion.json')
   values={p.parent.name:load(p)['elapsed_seconds'] for p in (folder/'probabilities').glob('*/complete.json')}
   values['fusion']=completion['fusion_seconds']; values['whole']=completion['elapsed_seconds']
   values['parent_startup_and_component_gaps']=values['whole']-sum(v for k,v in values.items() if k!='whole')
   rows.append({'run':name,'seconds':values})
  groups[kind]={'runs':rows,'medians':{key:statistics.median(row['seconds'][key] for row in rows) for key in rows[0]['seconds']} if rows else {}}
 result['groups'][tag]=groups
 telemetry=ROOT/f'{tag}_telemetry.jsonl'
 if telemetry.exists():
  samples=[json.loads(line) for line in telemetry.read_text().splitlines() if line.strip()]
  for group in groups.values():
   for row in group['runs']:
    audit=load(ROOT/'runs'/row['run']/'run_audit.json')
    rows=[v for v in samples if audit['started_unix']<=datetime.fromisoformat(v['utc']).timestamp()<=audit['finished_unix']]
    if rows:
     row['environment']={'samples':len(rows),'host_load1_median':statistics.median(float(v['loadavg'].split()[0]) for v in rows),'gpu_utilization_median_percent':statistics.median(float(v['gpu7'].split(',')[1]) for v in rows),'gpu_memory_peak_mib':max(float(v['gpu7'].split(',')[0]) for v in rows),'note':'Whole-run samples include loading, CPU preprocessing, fusion and gaps; not isolated CUDA kernel utilization.'}

(ROOT/'component_timing_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
