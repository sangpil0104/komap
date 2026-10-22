from pathlib import Path
import json,hashlib,zipfile
R=Path(__file__).resolve().parent
names=['infer_compact.py','infer_expert.py','zcdf_input.py','verify_package.py','selected_config.json','selected_R_lite16_N3.yaml','selected_assets.json','assembly.json','README.md','MODEL_STRUCTURE.md','ADOPTED_BASELINE.json','valid20_comparison.json','test10_comparison.json','completion.json','build_completion.json']
files=[R/n for n in names]
for directory in ['runtime','parent_metadata','parent_package','assets']:
 files.extend(p for p in (R/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pt','.pth','.npy','.npz','.pyc','.pkl'])
assert all(p.exists() for p in files)
checks={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(R/'CODE_CHECKSUMS.json').write_text(json.dumps(checks,indent=2)+'\n');files.append(R/'CODE_CHECKSUMS.json')
out=R/'Model349_lite14_N3_HRtrain70seed5_code.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(set(files)):z.write(p,'Model349_lite14_N3_HRtrain70seed5_code/'+str(p.relative_to(R)))
assert not any(x.endswith(('.pt','.pth')) for x in zipfile.ZipFile(out).namelist())
print(out,out.stat().st_size)
