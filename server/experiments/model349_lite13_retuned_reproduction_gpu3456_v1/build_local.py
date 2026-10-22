"""Apply documented local integration only to this new experiment's source copy."""
from pathlib import Path
import ast, difflib, hashlib, json, shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
OLD = ROOT.parent / 'model349_lite20_b4ws_retrain_gpu234567_v1'

def replace(path, old, new):
    text = path.read_text()
    assert old in text, (path, old[:100])
    path.write_text(text.replace(old, new))

def copy_function(path, old_path, name):
    text = path.read_text(); prior = old_path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    old_node = next(n for n in ast.parse(prior).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(True)
    lines[node.lineno-1:node.end_lineno] = [ast.get_source_segment(prior, old_node) + '\n']
    path.write_text(''.join(lines))

def main():
    shutil.copytree(OLD/'data', ROOT/'data')
    manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'data').rglob('*') if p.is_file()}
    (ROOT/'data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    assert manifest == {k:v for k,v in json.loads((OLD/'input_manifest.json').read_text()).items() if k.startswith('data/')}
    exp = SOURCE/'training/train_route/experiments'
    shutil.copy2(OLD/'source/experiments/local_image_metrics.py', exp/'local_image_metrics.py')
    functions = {
        'model1_unet_resnet34.py':['evaluate'], 'improve_five_causes.py':['metrics_from_predictions'],
        'calibrate_al.py':['score_confusions'], 'restart_ensemble.py':['metrics'],
        'encoder_structure_search.py':['metrics'], 'model181_raw_zcdf_aspp.py':['score'],
        'advanced_inference.py':['metrics'], 'hard_negative_finetune.py':['evaluate_calibrated']}
    for name, funcs in functions.items():
        for fun in funcs:
            copy_function(exp/'historical_scripts'/name, OLD/'source/experiments/historical_scripts'/name, fun)
    for path in (exp/'historical_scripts').glob('*.py'):
        text = path.read_text().replace('global_miou', 'image_miou_compat').replace('global_iou','image_class_iou')
        # Keep the old consumers' dictionary schema but all aliases now use the project metric.
        if '_local_progress' not in text or 'from local_image_metrics import progress' not in text:
            tree=ast.parse(text); body=tree.body; offset=0
            for n in body:
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value,str): offset=n.end_lineno
                elif isinstance(n, ast.ImportFrom) and n.module=='__future__': offset=n.end_lineno
                else: break
            lines=text.splitlines(True);lines.insert(offset,'from local_image_metrics import progress as _local_progress\n');text=''.join(lines)
        # Pure progress instrumentation: preserve iteration order and RNG draws.
        tree=ast.parse(text); lines=text.splitlines(True); changes=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.For):
                expression=ast.get_source_segment(text,n.iter)
                if isinstance(n.iter,ast.Name) and n.iter.id in ['loader','train_loader']:
                    changes.append((n.iter.lineno-1,n.iter.col_offset,n.iter.end_col_offset,f"_local_progress({expression}, '{path.stem}')"))
        for line,start,end,value in sorted(changes,reverse=True):lines[line]=lines[line][:start]+value+lines[line][end:]
        path.write_text(''.join(lines))
    # OOF prediction needs probabilities; do not compute a pooled/global score.
    path=SOURCE/'training/Model349_fusion805/model397_reproduction_bundle/model392/evaluate_image_mean.py'
    text=path.read_text(); text=text.replace('    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)\n','')
    start=text.index('            bins = target.reshape(-1)');end=text.index('        rows.append(row)',start);text=text[:start]+text[end:]
    start=text.index('        intersection = np.diag(pooled)');end=text.index('    (output /',start);text=text[:start]+text[end:]
    path.write_text(text)
    # Training logs for HRNet and F include elapsed time and ETA.
    for path in [exp/'model336_hrnet_best/train.py',SOURCE/'training/Model349_fusion805/model397_reproduction_bundle/model392/train.py']:
        replace(path,'for batch_index, (img, mask) in enumerate(train_loader):',
                'for batch_index, (img, mask) in enumerate(__import__("tqdm").tqdm(train_loader, desc=f"epoch {epoch}", mininterval=10)):')
    path=SOURCE/'training/refiner_retrain/train_refiner.py'
    replace(path,'for logits, gray, target in loader:',
            'for logits, gray, target in __import__("tqdm").tqdm(loader, desc=f"F epoch {epoch}/8", mininterval=10):')
    shell=SOURCE/'run_retrain.sh';text=shell.read_text()
    start=text.index('stage(){');end=text.index('\n\n# ---',start)
    text=text[:start]+'''stage(){ local name=$1 done=$2; shift 2
  ( cd "${CWD:-$ROOT}" && "$PY" "$ROOT/../stage.py" "$name" "$done" "$@" ); }
'''+text[end:]
    text=text.replace('--selection-metric image_mean_miou_exclude','--selection-metric image_mean_miou')
    text=text.replace('[[ -e $OUT/data_zcdf_png/done ]] || "$PY" "$ROOT/tools/make_zcdf_png.py"',
                      'stage prepare_n3 "$OUT/data_zcdf_png/done" "$ROOT/tools/make_zcdf_png.py"')
    start=text.index('prep_splits(){');end=text.index('\n\n# ---',start)
    text=text[:start]+'''prep_splits(){ stage prepare_splits "$E/splits/model397_split_manifest.json" "$B/scripts/prepare_oof_splits.py" --data-root "$DATA" --output-root "$E/splits"; }
'''+text[end:]
    # Discover the actual fixed recipe without executing any training or prediction.
    marker='prep_splits\nLANES='
    assert marker in text
    text=text.replace(marker,'''if [[ ${PLAN_ONLY:-0} == 1 ]]; then
  prep_splits; lane_resnet; lane_hrnet; lane_b4; lane_r326; lane_hr392 1; lane_hr392 2; lane_hr70
else
prep_splits
LANES=''',1)
    text=text.replace('# ---------------------------------------------------------------- F1:', 'fi\n\n# ---------------------------------------------------------------- F1:',1)
    text=text.replace('[[ -f $OUT/F/features/manifest.json ]] || "$PY"', 'stage F_features "$OUT/F/features/manifest.json"')
    text=text.replace(' > "$L/F_features.log" 2>&1','')
    text=text.replace('[[ -f $OUT/F/training/epoch8.pt ]] || "$PY"','stage F_train "$OUT/F/training/epoch8.pt"')
    text=text.replace(' > "$L/F_train.log" 2>&1','')
    text=text.replace('M=$OUT/model; rm -rf "$M";', '[[ ${PLAN_ONLY:-0} == 1 ]] && exit 0\nM=$OUT/model; [[ ! -e $M ]] || { echo "Refusing to overwrite $M"; exit 1; };')
    text=text.replace('  [[ -e $OUT/pred_$split/completion.json ]] && continue; rm -rf "$OUT/pred_$split"\n','')
    text=text.replace('ALLOW_RETRAINED=1 bash "$ROOT/run_predict.sh" "$M/weights" "$DATA/$split/images" "$OUT/pred_$split" "${G[0]}"',
                      '"$PY" "$ROOT/../timed_predict.py" --source "$ROOT" --weights "$M/weights" --images "$DATA/$split/images" --output "$OUT/pred_$split" --gpus "${1:-0}" --split "$split"')
    text=text.replace('[[ -d $DATA/valid/masks ]] && "$PY" "$ROOT/tools/score_masks.py"','stage score_valid "$OUT/pred_valid/metrics.json" "$ROOT/tools/score_masks.py"')
    shell.write_text(text)
    # Record R/E/F subprocess elapsed times; process exit includes CUDA completion and output persistence.
    p=SOURCE/'inference/infer_compact.py'
    replace(p,'import argparse,json,subprocess,sys,os,hashlib,shutil','import argparse,json,subprocess,sys,os,hashlib,shutil,time')
    replace(p,' def run(script,*args,env=None):subprocess.run([sys.executable,str(script),*map(str,args)],check=True,env=env)',
    ''' phase_times=[]
 def run(script,*args,env=None):
  start=time.perf_counter()
  subprocess.run([sys.executable,str(script),*map(str,args)],check=True,env=env)
  phase_times.append({'script':Path(script).name,'wall_seconds':time.perf_counter()-start})
  (a.output/'phase_timings.json').write_text(json.dumps(phase_times,indent=2)+'\\n')''')
    # Save differences against the untouched upstream, including new files.
    diffs=[]
    for p in sorted(SOURCE.rglob('*')):
        if p.is_file():
            q=ROOT/'upstream/lite13_retuned_reproduction'/p.relative_to(SOURCE)
            if q.exists() and p.read_bytes()!=q.read_bytes():
                diffs.extend(difflib.unified_diff(q.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='upstream/'+str(p.relative_to(SOURCE)),tofile='source/'+str(p.relative_to(SOURCE))))
    (ROOT/'source_changes.diff').write_text(''.join(diffs))

if __name__=='__main__':main()
