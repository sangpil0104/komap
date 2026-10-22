"""Run the packaged, hash-verified lite13 model in an activated conda komap environment."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--gpus',default='0',help='Physical GPU indices or UUIDs, e.g. 0 or 3,4,5,6')
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    if Path(sys.prefix).name!='komap':p.error('Activate conda environment komap before running this program.')
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'files_sha256.json').read_text())
    print('Verifying original model/code hashes...',flush=True)
    for name,digest in manifest.items():
        path=root/name
        if not path.is_file() or sha(path)!=digest:raise ValueError(f'File changed or missing: {name}')
    if len(list((root/'weights').rglob('*.pt')))!=13:raise ValueError('Expected exactly 13 weight files')
    print(f'PASS: {len(manifest)} original files, 13 model weights.',flush=True)
    if a.verify_only:return
    if a.images is None or a.output is None:p.error('--images and --output are required for inference')
    if not a.images.is_dir() or a.output.exists():p.error('Use an existing image directory and a new output directory')
    env=dict(os.environ,PYTHON=sys.executable,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',ALLOW_RETRAINED='1')
    subprocess.run(['bash',str(root/'code/run_predict.sh'),str(root/'weights'),str(a.images.resolve()),str(a.output.resolve()),a.gpus],check=True,env=env)

if __name__=='__main__':main()
