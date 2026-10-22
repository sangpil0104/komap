"""Local fixed-recipe training and D4 inference for the four Model326 folds."""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import components326
from runtime import ROOT, DATA, common, read_json, write_json, fold_pairs, image_key, metric, pairs, export_results


def save_probs(path, probs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        # Match upstream predict.py's probability archive precision.
        np.savez_compressed(handle, **{k: np.asarray(v, dtype=np.float16) for k, v in probs.items()})
    temp.replace(path)


def predict_items(members, items, label):
    values, records = {}, []
    for ip, mp in tqdm(items, desc=label, unit="image", mininterval=5):
        gray = common.load_gray(ip)
        probs = common.predict_probs(members, gray, tta=True, device="cuda").cpu().numpy()
        if not np.isfinite(probs).all() or not np.allclose(probs.sum(0), 1, atol=2e-3):
            raise ValueError(f"Invalid inference probabilities: {ip}")
        values[image_key(ip)] = probs
        records.append({"image": Path(ip).name, **metric(probs.argmax(0), common.load_mask(mp))})
    return values, {"image_miou": float(np.mean([r["image_miou"] for r in records])),
                    "empty_union_policy": "zero", "images": records}


def train_fold(run, fold, data=DATA):
    """Called only by explicit user-started --execute orchestration."""
    run = Path(run)
    config = read_json(run / "config.json")
    record = next(r for r in read_json(run / "folds.json")["completed_folds"] if r["fold"] == fold)
    out = run / f"fold{fold}"
    out.mkdir(exist_ok=False)
    config.update(data=str(data), out=str(out), train_splits=f"train_fold{fold}")
    write_json(out / "args.json", config)
    torch.set_num_threads(4)
    import cv2
    cv2.setNumThreads(1)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    components326.configure(config)
    dataset = components326.CropDS(fold_pairs(data, record), config["iters"] * config["bs"])
    workers = config["workers"]
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config["bs"], num_workers=workers, pin_memory=True,
        worker_init_fn=components326.worker_init, persistent_workers=workers > 0,
        multiprocessing_context="fork" if workers else None, drop_last=True)
    model = common.build_model(config["arch"], config["encoder"], pretrained=True).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    steps = config["iters"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / 300) * .5 * (1 + math.cos(math.pi * min(it, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", init_scale=config["amp_init_scale"])
    started, smooth, skipped = time.monotonic(), 0.0, 0
    bar = tqdm(loader, total=steps, desc=f"Model326 fold{fold}", unit="step", mininterval=10)
    for step, (x, y) in enumerate(bar, 1):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y) + config["dice_w"] * components326.dice_loss(logits.float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in fold {fold}, step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped += int(scaler.get_scale() < before)
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        smooth = .98 * smooth + .02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0 or step == 1 or step == steps:
            elapsed = time.monotonic() - started
            status = {"fold": fold, "step": step, "total_steps": steps, "loss_ema": smooth,
                      "lr": scheduler.get_last_lr()[0], "amp_scale": scaler.get_scale(),
                      "skipped_amp_steps": skipped, "elapsed_seconds": elapsed,
                      "eta_seconds": elapsed / step * (steps-step), "phase": "training"}
            write_json(out / "progress.json", status)
            with (out / "history.jsonl").open("a") as handle:
                import json
                handle.write(json.dumps(status) + "\n")
            bar.set_postfix(loss=f"{smooth:.4f}", lr=f"{status['lr']:.2e}")
    del loader, dataset, x, y, logits, loss
    tmp = out / "last.pt.tmp"
    torch.save(model.state_dict(), tmp)
    tmp.replace(out / "last.pt")
    del opt, scheduler, scaler
    torch.cuda.empty_cache()
    write_json(out / "progress.json", {"fold": fold, "step": steps, "phase": "oof_inference"})
    member = common.make_member(model.eval(), config["norm"], config["scale"], config["input"], config["feat_stats"])
    probs, metrics = predict_items([member], fold_pairs(data, record, True), f"fold{fold} OOF D4")
    save_probs(out / "oof_probabilities.npz", probs)
    write_json(out / "oof_metrics.json", metrics)
    write_json(out / "progress.json", {"fold": fold, "step": steps, "phase": "complete",
                                       "elapsed_seconds": time.monotonic() - started,
                                       "skipped_amp_steps": skipped})


def aggregate_oof(run, data=DATA):
    run = Path(run)
    expected = {image_key(ip): mp for ip, mp in pairs(data, "train")}
    values = {}
    for fold in range(1, 5):
        with np.load(run / f"fold{fold}/oof_probabilities.npz", allow_pickle=False) as archive:
            for key in archive.files:
                if key in values or key not in expected:
                    raise ValueError(f"Repeated/unknown OOF prediction: {key}")
                values[key] = archive[key]
    if set(values) != set(expected):
        raise ValueError("OOF predictions must cover each of the 70 training images exactly once")
    records = [{"image": key, **metric(a.argmax(0), common.load_mask(expected[key]))}
               for key, a in values.items()]
    save_probs(run / "oof/probabilities.npz", values)
    write_json(run / "oof/metrics.json", {
        "image_miou": float(np.mean([r["image_miou"] for r in records])),
        "empty_union_policy": "zero", "images": records})


def evaluate(run, data=DATA):
    run = Path(run).resolve()
    members = [common.load_member(str(run / f"fold{fold}")) for fold in range(1, 5)]
    values, metrics = predict_items(members, pairs(data, "valid"), "Model326 valid 4-fold D4")
    # A fresh evaluation directory preserves earlier probabilities and reports on re-evaluation.
    from datetime import datetime, timezone
    out = run / "evaluations" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    save_probs(out / "probabilities.npz", values)
    write_json(out / "metrics.json", metrics)
    result = export_results("lhs_Model326", {k: v.argmax(0).astype(np.uint8) for k, v in values.items()}, data,
                            {"run": str(run), "probabilities": str(out / "probabilities.npz"),
                             "config": read_json(run / "config.json"), "members": 4, "tta": "D4",
                             "selection": f"fixed final step {read_json(run / 'config.json')['iters']} per fold; no validation checkpoint selection"})
    write_json(out / "result.json", {"result_directory": str(result)})
    return result
