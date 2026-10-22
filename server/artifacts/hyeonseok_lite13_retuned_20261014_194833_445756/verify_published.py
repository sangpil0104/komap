"""Confirm remote commit and downloadable LFS objects without exposing credentials or signed URLs."""
import base64,json,subprocess
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parent;REPOSITORY='CNUKoMaP/giveItToHyeonSeok';GH='/home/sppark/.local/bin/gh'

def api(path):return json.loads(subprocess.check_output([GH,'api',path],text=True))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT/'repository',text=True).strip()

def main():
    prepared=json.loads((ROOT/'prepared_commit.json').read_text());commit=git('rev-parse','HEAD')
    assert api(f'repos/{REPOSITORY}/git/ref/heads/main')['object']['sha']==commit
    remote=api(f'repos/{REPOSITORY}/git/trees/{commit}?recursive=1');assert not remote.get('truncated')
    local={}
    for line in git('ls-tree','-r','HEAD').splitlines():
        meta,name=line.split('\t',1);local[name]=meta.split()[2]
    assert local=={x['path']:x['sha'] for x in remote['tree'] if x['type']=='blob'}
    changes=api(f'repos/{REPOSITORY}/commits/{commit}')
    assert changes['parents'][0]['sha']==prepared['parent']
    assert {r['filename'] for r in changes['files']}==set(prepared['files'])
    old=json.loads(git('show',prepared['parent']+':manifest.json'))
    current=json.loads((ROOT/'repository/manifest.json').read_text())
    for name,value in old['files'].items():
        assert current['files'][name]==value
        assert git('rev-parse',f'HEAD:{name}')==git('rev-parse',f'{prepared["parent"]}:{name}')
    checkpoints=json.loads((ROOT/'repository/metadata/checkpoints.json').read_text())
    assert len(checkpoints)==72 and checkpoints[:59]==json.loads(git('show',prepared['parent']+':metadata/checkpoints.json'))
    objects=[dict(oid=v['sha256'],size=v['bytes']) for name,v in current['files'].items() if name.startswith('models/')]
    assert len(objects)==30
    token=subprocess.check_output([GH,'auth','token','--hostname','github.com'],text=True).strip()
    request=Request(f'https://github.com/{REPOSITORY}.git/info/lfs/objects/batch',
        data=json.dumps(dict(operation='download',transfers=['basic'],objects=objects,ref={'name':'refs/heads/main'})).encode(),method='POST',
        headers={'Authorization':'Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
                 'Content-Type':'application/vnd.git-lfs+json','Accept':'application/vnd.git-lfs+json','User-Agent':'komap-lite13-verify'})
    with urlopen(request,timeout=30) as f:results=json.load(f)['objects']
    found={r['oid']:r for r in results};assert len(found)==30
    for row in objects:
        r=found[row['oid']];assert r['size']==row['size'] and not r.get('error')
        assert r.get('actions',{}).get('download',{}).get('href')
    action=found[prepared['sha256']]['actions']['download']
    try:
        headers=dict(action.get('header',{}));headers['Range']='bytes=0-3'
        with urlopen(Request(action['href'],headers=headers),timeout=30) as f:
            signature=f.read(4);http_status=f.status
        assert signature==b'PK\x03\x04'
    except Exception:
        raise RuntimeError('Remote model ZIP download check failed; credentials and signed URLs omitted.') from None
    report=dict(ok=True,repository=REPOSITORY,commit=commit,previous_59_checkpoints_unchanged=True,
        checkpoint_count=72,model_archives=30,new_checkpoints=13,all_lfs_objects_downloadable=True,new_zip_download_signature_verified=True,
        bytes=prepared['bytes'],sha256=prepared['sha256'],http_status=http_status,
        archive_url=f'https://github.com/{REPOSITORY}/blob/main/{prepared["archive"]}',
        report_url=f'https://github.com/{REPOSITORY}/blob/main/reports/model_results_20261014_1/model_results_20261014_1.md')
    (ROOT/'published_repository.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
