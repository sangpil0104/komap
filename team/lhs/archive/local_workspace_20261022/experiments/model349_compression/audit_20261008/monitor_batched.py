import json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
root=Path('/workspace/model349_compress/audit_20261008')
with (root/'batched_telemetry.jsonl').open('x') as out:
    for _ in range(1440):
        if (root/'batched_complete.json').exists(): break
        row={'utc':datetime.now(timezone.utc).isoformat(),'gpu7':subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],text=True).strip(),'compute_apps':subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip(),'loadavg':Path('/proc/loadavg').read_text().strip()}
        out.write(json.dumps(row)+'\n'); out.flush(); time.sleep(5)
