"""Verify GitHub file blobs, LFS object availability, and previous ZIP preservation."""
import base64
import json
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
REPO = 'CNUKoMaP/sppark'
GH = '/home/sppark/.local/bin/gh'


def api(path):
    return json.loads(subprocess.check_output([GH, 'api', path], text=True))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT/'repository', text=True).strip()


def main():
    prepared = json.loads((ROOT/'commit_prepared.json').read_text())
    commit = git('rev-parse', 'HEAD')
    remote_commit = api(f'repos/{REPO}/commits/{commit}')
    assert remote_commit['parents'][0]['sha'] == prepared['parent']
    assert {f['filename'] for f in remote_commit['files']} == set(prepared['files'])
    for change in remote_commit['files']:
        assert change['status'] == ('modified' if change['filename']=='.gitattributes' else 'added')
    for filename in prepared['files']:
        expected_blob = git('rev-parse', f'HEAD:{filename}')
        remote = api(f'repos/{REPO}/contents/{filename}?ref=main')
        assert remote['sha'] == expected_blob, filename
    for archive in prepared['archives']:
        filename = archive['filename']
        blob = api(f'repos/{REPO}/git/blobs/{git("rev-parse", f"HEAD:{filename}")}')
        pointer = base64.b64decode(blob['content']).decode()
        assert f'oid sha256:{archive["sha256"]}\n' in pointer
        assert f'size {archive["bytes"]}\n' in pointer
    previous = 'model349_lite20_seeds3407-3414_weights.zip'
    assert git('rev-parse', f'HEAD:{previous}') == git('rev-parse', f'{prepared["parent"]}:{previous}')
    token = subprocess.check_output([GH, 'auth', 'token', '--hostname', 'github.com'], text=True).strip()
    objects = [dict(oid=r['sha256'], size=r['bytes']) for r in prepared['archives']]
    body = dict(operation='download', transfers=['basic'], objects=objects, ref={'name':'refs/heads/main'})
    request = Request(f'https://github.com/{REPO}.git/info/lfs/objects/batch',
        data=json.dumps(body).encode(), method='POST', headers={
            'Authorization':'Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
            'Content-Type':'application/vnd.git-lfs+json', 'Accept':'application/vnd.git-lfs+json',
            'User-Agent':'komap-lfs-verify'})
    with urlopen(request, timeout=30) as response:
        returned = json.load(response)['objects']
    assert len(returned)==3
    by_oid = {o['oid']:o for o in returned}
    for expected in objects:
        actual = by_oid[expected['oid']]
        assert actual['size']==expected['size'] and not actual.get('error')
        assert actual.get('actions',{}).get('download',{}).get('href')
    # Never save signed download URLs, temporary headers, or the authentication token.
    report = dict(ok=True, repository=REPO, commit=commit, files=prepared['files'],
                  previous_seed8_zip_unchanged=True, lfs_objects_download_available=True,
                  archives=[dict(**r, url=f'https://github.com/{REPO}/blob/main/{r["filename"]}') for r in prepared['archives']])
    (ROOT/'published_repository.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
