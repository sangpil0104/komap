"""Verify downloaded model archives with Python's standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--best',action='store_true',help='Verify only the best new candidate per family')
    parser.add_argument('--family',choices=['HRNet347','B4-326','B4-322','B4-warmstart','HRNet392','B4-329','Lite13-retuned'])
    parser.add_argument('--deep',action='store_true',help='Also decompress and hash the original checkpoints')
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text())
    candidates=json.loads((root/'metadata/all_candidates.json').read_text())
    if args.best:
        best=json.loads((root/'metadata/best_per_family.json').read_text())
        candidates=list(best.values())
        if args.family in ['B4-329','Lite13-retuned']:
            parser.error('This model is fixed and has no seed ranking; omit --best for this family.')
    else:
        extra=root/'metadata/additional_models.json'
        if extra.exists():candidates+=json.loads(extra.read_text())
        complete=root/'metadata/complete_models.json'
        if complete.exists():candidates+=json.loads(complete.read_text())
    if args.family:candidates=[r for r in candidates if r['family']==args.family]
    assert candidates,'No matching candidates'
    files={r['archive'] for r in candidates}|{'source_snapshot.zip'}
    for index,name in enumerate(sorted(files),1):
        path=root/name; expected=manifest['files'][name]
        if not path.is_file() or path.stat().st_size!=expected['bytes']:
            raise SystemExit(f'Missing/incomplete ZIP: {name}. Download with git lfs pull or Download raw file.')
        if file_sha(path)!=expected['sha256']:raise SystemExit(f'SHA256 mismatch: {name}')
        print(f'[{index}/{len(files)}] OK {name}',flush=True)
    count=0
    for candidate in candidates:
        if args.deep:
            with ZipFile(root/candidate['archive']) as archive:
                for checkpoint in candidate['checkpoints']:
                    with archive.open(checkpoint['member']) as stream:
                        digest=hashlib.file_digest(stream,'sha256').hexdigest()
                    if digest!=checkpoint['checkpoint_sha256']:
                        raise SystemExit(f'Checkpoint mismatch: {checkpoint["task_id"]}')
        count+=len(candidate['checkpoints'])
    print(f'PASS: {len(candidates)} packages, {count} checkpoints; deep={args.deep}')


if __name__=='__main__':main()
