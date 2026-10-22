"""Build-time manifest writer; never runs on training startup or resume."""
from common import ROOT,BASE,MODEL,sha,write_json,read_json

def main():
    inputs={}
    def add(root):
        for p in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if p.is_file():inputs[str(p)]=sha(p)
    for d in ['data','folds','expert_splits']:add(ROOT/d)
    for d in ['release_inference','expert_oof','expert_valid','stage2','refiner_features/train']:
        add(BASE/d)
    add(MODEL)
    for name in ['outputs/model109_aspp_native/best.pt','outputs/model257_full_image_stats_1x/best.pt',
                 'outputs/model336_hrnet_best/last.pt','outputs/restart51_aspp_768_finetune/best.pt',
                 'expert/refiner/completion.json','expert/refiner/recipe.json']:
        add(BASE/name)
    for f in range(1,5):
        add(BASE/f'expert/model329/fold{f}/last.pt');add(BASE/f'expert/model329/fold{f}/args.json')
    write_json(ROOT/'input_manifest.json',inputs)
    code={}
    for d in ['source','frozen','fusion_source','compact_source']:
        for p in sorted((ROOT/d).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':code[str(p.relative_to(ROOT))]=sha(p)
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.suffix in ['.py','.sh','.json','.md'] and p.name not in ['code_manifest.json','input_manifest.json','BUILD_REPORT.md']:
            code[p.name]=sha(p)
    write_json(ROOT/'code_manifest.json',code)
    print(f'Sealed {len(code)} code files and {len(inputs)} immutable input files.')

if __name__=='__main__':main()
