"""Build-time manifest generation; never run automatically on resume."""
from common import ROOT,BASE,MODEL,sha,write_json

def main():
    inputs={}
    for folder in [ROOT/'data',MODEL]:
        for p in sorted(folder.rglob('*')):
            if p.is_file():inputs[str(p)]=sha(p)
    p=BASE/'stage2/metrics.json';inputs[str(p)]=sha(p)
    write_json(ROOT/'input_manifest.json',inputs)
    code={}
    for folder in ['source','frozen','fusion_source','compact_source']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':code[str(p.relative_to(ROOT))]=sha(p)
    for p in ROOT.iterdir():
        if p.is_file() and p.name not in ['input_manifest.json','code_manifest.json','BUILD_REPORT.md']:
            code[p.name]=sha(p)
    write_json(ROOT/'code_manifest.json',code)
    print(f'Sealed {len(code)} source files and {len(inputs)} input files.')

if __name__=='__main__':main()
