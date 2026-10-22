"""Verify the committed Git tree and server availability of all 28 LFS ZIPs."""
import base64
import json
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parent
GH='/home/sppark/.local/bin/gh'
REPO='CNUKoMaP/giveItToHyeonSeok'


def api(path):
    return json.loads(subprocess.check_output([GH,'api',path],text=True))


def main():
    local=ROOT/'repository'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=local,text=True).strip()
    remote=api(f'repos/{REPO}/git/trees/{commit}?recursive=1')
    assert not remote.get('truncated')
    expected={}
    for entry in subprocess.check_output(['git','ls-tree','-r','HEAD'],cwd=local,text=True).splitlines():
        header,path=entry.split('\t',1);expected[path]=header.split()[2]
    actual={row['path']:row['sha'] for row in remote['tree'] if row['type']=='blob'}
    assert actual==expected
    assert api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']==commit
    manifest=json.loads((local/'manifest.json').read_text())
    objects=[]
    for name,row in manifest['files'].items():
        if not name.startswith('models/'):continue
        pointer=subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=local,text=True)
        assert f'oid sha256:{row["sha256"]}\n' in pointer
        assert f'size {row["bytes"]}\n' in pointer
        objects.append(dict(oid=row['sha256'],size=row['bytes']))
    assert len(objects)==28
    token=subprocess.check_output([GH,'auth','token','--hostname','github.com'],text=True).strip()
    request=Request(f'https://github.com/{REPO}.git/info/lfs/objects/batch',
        data=json.dumps(dict(operation='download',transfers=['basic'],objects=objects,ref={'name':'refs/heads/main'})).encode(),
        method='POST',headers={
            'Authorization':'Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
            'Content-Type':'application/vnd.git-lfs+json','Accept':'application/vnd.git-lfs+json',
            'User-Agent':'komap-handoff-verify'})
    with urlopen(request,timeout=45) as response:result=json.load(response)
    indexed={o['oid']:o for o in result['objects']}
    assert len(indexed)==28
    for obj in objects:
        value=indexed[obj['oid']]
        assert value['size']==obj['size'] and not value.get('error')
        assert value.get('actions',{}).get('download',{}).get('href')
    # Temporary download URLs and authentication are deliberately not persisted.
    best=json.loads((local/'metadata/best_per_family.json').read_text())
    receipt=dict(ok=True,repository=REPO,commit=commit,checkpoint_count=56,archive_count=28,
                 repository_files=len(actual),all_lfs_objects_available=True,
                 readme_url=f'https://github.com/{REPO}#readme',
                 best={k:dict(id=v['id'],image_miou_percent=v['image_miou_percent'],
                       url=f'https://github.com/{REPO}/blob/main/{v["archive"]}') for k,v in best.items()})
    (ROOT/'published_repository.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
