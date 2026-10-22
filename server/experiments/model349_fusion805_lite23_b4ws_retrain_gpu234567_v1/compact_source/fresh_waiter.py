from pathlib import Path
import subprocess,time,json
r=Path('/workspace/model349_compress/fusion805_compression_round2_20261010');f=r/'fresh23';idle=0
while idle<3:
 values=[subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip() for g in [6,7]]
 available=all(int(v.split(',')[0])<100 and int(v.split(',')[1])==0 for v in values)
 idle=idle+1 if available else 0
 (f/'waiting.json').write_text(json.dumps({'status':'waiting_for_idle_gpu' if idle<3 else 'idle_confirmed','gpu6_gpu7':values,'idle_observations':idle,'time_unix':time.time()}));time.sleep(15)
subprocess.run(['/workspace/venv-komap/bin/python',str(f/'supervisor_retry.py')],check=True)
subprocess.run(['/workspace/venv-komap/bin/python',str(r/'fresh_postprocess.py')],check=True)
