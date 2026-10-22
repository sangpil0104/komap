from pathlib import Path
import json,os,subprocess
ROOT=Path(__file__).resolve().parent;REPO=ROOT/'repository'
LFS=Path('/Data3/sppark/komap/artifacts/model349_lite20_seed8_weights_20261011_193850_048767/repository_upload_20261011_195518_472355/tools')
env=dict(os.environ,PATH=str(LFS)+os.pathsep+os.environ['PATH'])
def run(*args):return subprocess.run(['git',*args],cwd=REPO,env=env,check=True)
def get(*args):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()
build=json.loads((ROOT/'build_verified.json').read_text());assert get('rev-parse','HEAD')==build['parent']
run('lfs','install','--local')
run('config','--local','credential.helper','!/home/sppark/.local/bin/gh auth git-credential')
run('config','--local','user.name','sangpil0104');run('config','--local','user.email','psp0104@naver.com')
files=['README.md','SHA256SUMS.txt','manifest.json','metadata/checkpoints.csv','metadata/checkpoints.json',
       'verify_download.py','metadata/complete_models.json',build['archive']]
files += [str(p.relative_to(REPO)) for p in sorted((REPO/'reports/model_results_20261014_1').iterdir()) if p.is_file()]
changed=set(get('diff','--name-only').splitlines())|set(get('diff','--cached','--name-only').splitlines())|set(get('ls-files','--others','--exclude-standard').splitlines())
assert changed==set(files),(changed,set(files))
run('add','--',*files)
assert int(get('cat-file','-s',':'+build['archive']))<200
pointer=get('show',':'+build['archive'])
assert 'oid sha256:'+build['sha256'] in pointer and 'size '+str(build['bytes']) in pointer
assert set(get('diff','--cached','--name-only').splitlines())==set(files)
run('-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
prepared={**build,'files':files}
(ROOT/'prepared_commit.json').write_text(json.dumps(prepared,indent=2)+'\n')
run('commit','-m','Add retrained lite13 retuned ensemble with weights, inference code and timings')
print('COMMIT',get('rev-parse','HEAD'))
