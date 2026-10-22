import json,os,subprocess,time
from pathlib import Path
r=Path('/workspace/model349_compress/base12_highflip4_20261008')
while not any((r/x).exists() for x in ['completion.json','failure.json']):
 data={'time':time.time(),'load_average':os.getloadavg()}
 for key,args in [('gpu',['--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']),('processes',['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])]:
  data[key]=subprocess.check_output(['nvidia-smi']+args,text=True)
 with (r/'monitor.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
 time.sleep(5)
