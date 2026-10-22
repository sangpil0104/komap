"""Lightweight queue: no numpy/torch imports or preparation during previous timings."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/model349_compress/base12_partial_distill329_20261008');PREVIOUS=ROOT.parent/'base12_singleworker_20261008'
while not any((PREVIOUS/n).exists() for n in ['completion.json','failure.json']):time.sleep(5)
if (PREVIOUS/'failure.json').exists():
 (ROOT/'skipped.json').write_text(json.dumps({'reason':'preceding worker failed; training not launched'},indent=2)+'\n')
else:
 with (ROOT/'run.log').open('x') as log:
  subprocess.run(['/workspace/venv-komap/bin/python','-B','-u',str(ROOT/'run_pilot.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
