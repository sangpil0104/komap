from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 m=json.load(open(R/'selected_assets.json'));assert m['count']==len(m['checkpoints'])==14
 for x in m['checkpoints']+m['metadata']:
  p=(R/'parent_package/Model349_exh14' if x['namespace']=='R' else R/'assets')/x['path'];assert sha(p)==x['sha256'],str(p)
 if (R/'CODE_CHECKSUMS.json').exists():
  for name,h in json.load(open(R/'CODE_CHECKSUMS.json')).items():assert sha(R/name)==h,name
 print('14 checkpoints, metadata, and packaged code SHA verified')
if __name__=='__main__':main()
