"""Verify remote contents, all model LFS objects, and previous models unchanged."""
import base64
import json
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parent
REPO='CNUKoMaP/giveItToHyeonSeok'
GH='/home/sppark/.local/bin/gh'


def api(path):
    return json.loads(subprocess.check_output([GH,'api',path],text=True))


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT/'repository',text=True).strip()


def main():
    prepared=json.loads((ROOT/'prepared_commit.json').read_text())
    commit=git('rev-parse','HEAD')
    assert api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']==commit
    tree=api(f'repos/{REPO}/git/trees/{commit}?recursive=1');assert not tree.get('truncated')
    expected={}
    for line in git('ls-tree','-r','HEAD').splitlines():
        head,name=line.split('\t',1);expected[name]=head.split()[2]
    assert expected=={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    changes=api(f'repos/{REPO}/commits/{commit}')
    assert changes['parents'][0]['sha']==prepared['parent']
    assert {r['filename'] for r in changes['files']}==set(prepared['files'])
    old=json.loads(git('show',prepared['parent']+':manifest.json'))
    current=json.loads((ROOT/'repository/manifest.json').read_text())
    for name,value in old['files'].items():
        assert current['files'][name]==value
        assert git('rev-parse',f'HEAD:{name}')==git('rev-parse',f'{prepared["parent"]}:{name}')
    objects=[dict(oid=r['sha256'],size=r['bytes']) for name,r in current['files'].items() if name.startswith('models/')]
    assert len(objects)==29
    token=subprocess.check_output([GH,'auth','token','--hostname','github.com'],text=True).strip()
    request=Request(f'https://github.com/{REPO}.git/info/lfs/objects/batch',
        data=json.dumps(dict(operation='download',transfers=['basic'],objects=objects,ref={'name':'refs/heads/main'})).encode(),
        method='POST',headers={
            'Authorization':'Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
            'Content-Type':'application/vnd.git-lfs+json','Accept':'application/vnd.git-lfs+json',
            'User-Agent':'komap-b4-329-verify'})
    with urlopen(request,timeout=30) as response:results=json.load(response)['objects']
    by_oid={o['oid']:o for o in results}
    assert len(by_oid)==29
    for obj in objects:
        value=by_oid[obj['oid']]
        assert value['size']==obj['size'] and not value.get('error')
        assert value.get('actions',{}).get('download',{}).get('href')
    report=dict(ok=True,repository=REPO,commit=commit,previous_56_weights_unchanged=True,
                repository_checkpoint_count=59,model_archives=29,all_lfs_downloads_available=True,
                added_folds=[1,3,4],added_training_seed=0,bytes=prepared['bytes'],sha256=prepared['sha256'],
                url=f'https://github.com/{REPO}/blob/main/{prepared["archive"]}',
                readme_url=f'https://github.com/{REPO}#readme')
    (ROOT/'published_repository.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
