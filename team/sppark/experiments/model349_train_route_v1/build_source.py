"""Reproducible local adaptation. Never edits the downloaded upstream tree."""
import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_body(text, name, body):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    start = node.body[0].lineno - 1
    lines[start:node.end_lineno] = ["    " + line + "\n" if line else "\n" for line in body.strip().splitlines()]
    return "".join(lines)


def exact(text, old, new):
    if old not in text:
        raise ValueError(f"Patch target missing: {old[:120]}")
    return text.replace(old, new)


def add_import(text, statement):
    tree = ast.parse(text)
    at = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            at = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            at = node.end_lineno
        else:
            break
    lines = text.splitlines(keepends=True)
    lines.insert(at, statement + "\n")
    return "".join(lines)


def progress_loops(text, label):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        iterable = node.iter
        value = ast.get_source_segment(text, iterable)
        total = None
        if isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name) and iterable.func.id == "enumerate":
            candidate = iterable.args[0]
        else:
            candidate = iterable
        candidate_text = ast.get_source_segment(text, candidate)
        selected = candidate_text in {"loader", "train_loader", "dl", "samples", "valid_samples", "valid_ds", "ds", "valid_data", "completed", "settings"}
        if isinstance(candidate, ast.Call) and isinstance(candidate.func, ast.Name):
            selected |= candidate.func.id == "range" and any(s in candidate_text for s in ("epoch", "args.steps", "len(rows)", "n_images", "image_count", "len(_WORKER_SETTINGS)"))
            selected |= candidate.func.id == "list_split"
        if isinstance(candidate, ast.Call) and isinstance(candidate.func, ast.Attribute):
            selected |= candidate.func.attr == "imap_unordered"
            if candidate.func.attr == "imap_unordered":
                total = "len(rows)"
        if not selected:
            continue
        if candidate_text == "completed":
            total = "n_images" if "n_images" in text else "image_count"
        if candidate_text == "dl":
            total = "args.iters"
        suffix = f", total={total}" if total else ""
        wrapped = f"_local_progress({candidate_text}, {label!r}{suffix})"
        start = offsets[candidate.lineno - 1] + candidate.col_offset
        end = offsets[candidate.end_lineno - 1] + candidate.end_col_offset
        edits.append((start, end, wrapped))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    if edits:
        text = add_import(text, "from local_image_metrics import progress as _local_progress")
    return text


def build():
    source = ROOT / "source"
    if source.exists():
        raise FileExistsError("source already exists; preserve it and build a new experiment version")
    shutil.copytree(ROOT / "upstream", source)
    changes = []
    for path in sorted(source.rglob("*.py")):
        text = original = path.read_text()
        rel = path.relative_to(source)
        # Shared historical metric implementations: retain consumer field names,
        # replacing global-field names with explicit image-metric compatibility aliases.
        funcs = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
        for node in funcs:
            segment = ast.get_source_segment(original, node)
            args = [a.arg for a in node.args.args]
            if args == ["predictions", "targets"] and "global_miou" in segment:
                text = replace_body(text, node.name,
                    "from local_image_metrics import summary\nreturn summary(predictions, targets, CLASS_NAMES)")
        if path.name == "model1_unet_resnet34.py":
            text = replace_body(text, "evaluate", '''
from local_image_metrics import summary
model.eval()
predictions, targets = [], []
total_loss = count = 0
for images, masks in loader:
    images, masks = images.to(device), masks.to(device)
    with autocast(enabled=use_amp):
        logits = model(images)
        loss = combined_loss(logits, masks)
    predictions.extend(logits.argmax(1).cpu().numpy())
    targets.extend(masks.cpu().numpy())
    total_loss += float(loss) * images.shape[0]
    count += images.shape[0]
result = summary(predictions, targets, CLASS_NAMES)
result["loss"] = total_loss / max(count, 1)
return result
''')
        if path.name == "improve_model6.py":
            text = replace_body(text, "evaluate_model", '''
from local_image_metrics import summary
model.eval()
predictions, targets = [], []
total_loss = count = 0
for images, masks in loader:
    images, masks = images.to(device), masks.to(device)
    with autocast(enabled=use_amp):
        logits = model(images)
        loss = segmentation_loss(logits, masks, loss_mode)
    probs = tta_logits(model, images, use_amp) if use_tta else torch.softmax(logits, 1)
    predictions.extend(probs.argmax(1).cpu().numpy())
    targets.extend(masks.cpu().numpy())
    total_loss += float(loss) * images.shape[0]
    count += images.shape[0]
result = summary(predictions, targets, CLASS_NAMES)
result.update(loss=total_loss / max(count, 1), tta=use_tta)
return result
''')
        if path.name == "hard_negative_finetune.py":
            text = replace_body(text, "evaluate_calibrated", '''
from local_image_metrics import summary
model.eval()
predictions, targets = [], []
for image_path, mask_path in samples:
    probabilities = predict_probabilities(model, image_path, args.patch_size, args.val_stride, device,
                                          use_amp, args.tile_batch_size, True)
    adjusted = np.log(np.maximum(probabilities, 1e-7))
    adjusted[3] += args.eutectic_bias
    predictions.append(adjusted.argmax(0).astype(np.uint8))
    targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
return summary(predictions, targets, CLASS_NAMES)
''')
        if path.name == "common.py" and "def image_miou(" in text:
            text = replace_body(text, "image_miou", '''
from local_image_metrics import class_ious
iou = class_ious(pred, gt)
return float(iou.mean()), float(iou.mean()), iou
''')
        if str(rel) == "experiments/model336_hrnet_best/train.py":
            text = exact(text, "conf = torch.zeros(NUM_CLASSES, NUM_CLASSES)", "image_ious = []")
            text = exact(text, 'conf += confusion_matrix(logits.argmax(1).cpu(), sample["mask"], NUM_CLASSES)',
                         'image_ious.append(iou_per_class(confusion_matrix(logits.argmax(1).cpu(), sample["mask"], NUM_CLASSES)))')
            text = exact(text, "return iou_per_class(conf)", "return torch.stack(image_ious).mean(0)")
        if path.name == "evaluate_image_mean.py":
            text = exact(text, "    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)\n", "")
            start = text.index("            bins = target.reshape")
            end = text.index("        rows.append(row)", start)
            text = text[:start] + text[end:]
            start = text.index("        intersection = np.diag(pooled)")
            end = text.index('    (output / f"{args.split}_summary.json")', start)
            text = text[:start] + text[end:]
        if path.name in ("model335_exhaustive_b4_route_search.py", "model341_model329_joint_route_search.py"):
            # Keep per-image confusion records for downstream consumers; never
            # pool them to calculate an IoU.
            marker = "        intersection = np.diag(matrix).astype(np.float64)"
            start = text.index(marker, text.index("    records = []"))
            end = text.index("        records.append({", start)
            text = text[:start] + '''        per_image = image_confusion[index]
        intersection = np.diagonal(per_image, axis1=1, axis2=2).astype(np.float64)
        union = per_image.sum(1) + per_image.sum(2) - intersection
        global_iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0).mean(0)
''' + text[end:]
        if path.name == "model335_exhaustive_b4_route_search.py":
            text = replace_body(text, "grouped_cv_global", 'raise RuntimeError("Pooled global evaluation is disabled by project policy")')
            text = exact(text, '        "grouped_cv_global": grouped_cv_global(records, image_confusion, names),\n', "")
            text = exact(text, '"grouped_cv_image_empty_zero", "grouped_cv_image_empty_skipped",\n        "grouped_cv_global")',
                         '"grouped_cv_image_empty_zero", "grouped_cv_image_empty_skipped")')
        if path.name == "model338_fine_gamma_route_search.py":
            text = exact(text, "    for class_name, field in gamma_fields:\n", '''    for class_name, field in gamma_fields:
        # A retrained parent can keep the original Al channel (no Al expert).
        # Its gamma must remain zero until an actual expert is selected.
        if field == "al_gamma" and "al_source" not in seed:
            continue
''')
        if path.name == "model339_joint_gamma_route_search.py":
            text = exact(text, "    combinations = list(itertools.product(*(GAMMA_GRID[key] for key in grid_keys)))", '''    active_grid = dict(GAMMA_GRID)
    if "al_source" not in strict_parent:
        active_grid["al_gamma"] = (0.0,)
    combinations = list(itertools.product(*(active_grid[key] for key in grid_keys)))''')
            text = exact(text, '    if len(settings) != 401:\n        raise AssertionError(f"expected baseline + 400 combinations, got {len(settings)}")',
                         '    if len(settings) != 1 + len(combinations):\n        raise AssertionError("Candidate grid count mismatch")')
            text = exact(text, '"full_grid_count_including_model270_reference": 401,',
                         '"full_grid_count_including_model270_reference": 1 + len(combinations),')
            text = exact(text, '"gamma_grid": GAMMA_GRID,', '"gamma_grid": active_grid,')
        if path.name == "model346_model336_into_341_route.py":
            text = exact(text, '    base_route = model339_result["best_image_empty_zero"]', '''    if baseline_record.get("base_model", 339) == 338:
        base_route = json.loads((outputs / "model338_fine_gamma_route_search/results.json")
                                .read_text(encoding="utf-8"))["best_image_empty_zero"]
    else:
        base_route = model339_result["best_image_empty_zero"]''')
            text = exact(text, '    print(json.dumps({key: value for key, value in result.items()', '''    if args.anchor_model346:
        from local_result_export import export_final
        def final_prediction(row):
            guide = model341.route.bilateral_weights(row["item"]["z"])
            return _candidate_prediction(row, settings[best], base_route, gamma329, guide,
                                         base_gamma336, base_gamma347)
        export_final(rows, names, final_prediction, root, result)
    print(json.dumps({key: value for key, value in result.items()''')
        text = text.replace("global_miou", "image_miou_compat").replace("global_iou", "image_class_iou")
        text = text.replace("best_global", "best_image_compat").replace("grouped_cv_global", "grouped_cv_disabled")
        text = text.replace("global={", "image_compat={")
        text = text.replace("plus pooled global mIoU", "image-metric compatibility fields only")
        text = text.replace("plus skip/global references", "local empty=0 selection")
        # All selection aliases use the same empty=0 score in adapted training.
        text = text.replace("--selection-metric image_mean_miou_exclude", "--selection-metric image_mean_miou")
        text = progress_loops(text, path.stem)
        ast.parse(text)
        if text != original:
            path.write_text(text)
            changes.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                           fromfile="upstream/" + str(rel), tofile="source/" + str(rel)))
    pipeline = source / "experiments/run_model349_pipeline.sh"
    text = original = pipeline.read_text()
    text = exact(text, 'export PYTHONPATH="$ROOT/experiments:', '''if [[ -n "${KOMAP_SETUP_ROOT:-}" ]]; then
  cp -a "$ROOT" "$RUN_ROOT/source_snapshot"
  ROOT="$RUN_ROOT/source_snapshot"
  cp "$KOMAP_SETUP_ROOT/provenance.json" "$RUN_ROOT/provenance.json"
  cp "$KOMAP_SETUP_ROOT/source_manifest.json" "$RUN_ROOT/source_manifest.json"
  cp "$KOMAP_SETUP_ROOT/reports/local_patches.diff" "$RUN_ROOT/local_patches.diff"
  cp "$KOMAP_SETUP_ROOT/datasets/manifest.json" "$RUN_ROOT/dataset_manifest.json"
  "$PY" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"
fi

export PYTHONPATH="$ROOT/experiments:''')
    text = text.replace("--selection-metric image_mean_miou_exclude", "--selection-metric image_mean_miou")
    # Preserve serial scheduling but expose each B4 training/inference tqdm in terminal.
    import re
    text = re.sub(r'> "(\$[^"\n]+\.log)" 2>&1', r'2>&1 | tee "\1"', text)
    pipeline.write_text(text)
    changes.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                   fromfile="upstream/experiments/run_model349_pipeline.sh", tofile="source/experiments/run_model349_pipeline.sh"))
    for adapter in (ROOT / "adapters").glob("*.py"):
        shutil.copy2(adapter, source / "experiments" / adapter.name)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "local_patches.diff").write_text("".join(changes))
    manifest = []
    for base in (ROOT / "upstream", source):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                manifest.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Adapted source ready: {source}; {len(manifest)} preserved/source files recorded")


if __name__ == "__main__":
    build()
