#!/usr/bin/env python3
"""Train a team-style representation branch with a 512->768->native schedule.

The earlier branch used only 512-pixel crops.  The team reports
attribute their strongest branch to a staged resolution schedule, so this
script keeps the same raw/z-score/CDF + Residual ASPP representation while
changing the crop size between stages.  Validation is always evaluated with a
fixed inference crop so checkpoints from different stages are comparable.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from raw_zcdf_aspp import (
    RepresentationPatchDataset,
    TeamRepresentationASPP,
    evaluate,
    load_checkpoint,
    seed_everything,
    train_epoch,
)
from improve_five_causes import collect_split


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4."))
         else decoder).append(parameter)
    return torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")

    stages = [
        (512, args.stage0_epochs, args.stage0_patches, args.stage0_batch),
        (768, args.stage1_epochs, args.stage1_patches, args.stage1_batch),
        (1024, args.stage2_epochs, args.stage2_patches, args.stage2_batch),
    ]
    total_epochs = sum(stage[1] for stage in stages)
    model = TeamRepresentationASPP(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = GradScaler(enabled=use_amp)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_state, best_record = -1.0, None, None
    history = []
    global_epoch = 0
    start_time = time.time()
    print(f"device={device} stages={stages} params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    for stage_index, (patch_size, epochs, patches_per_image, batch_size) in enumerate(stages):
        if epochs <= 0:
            continue
        dataset = RepresentationPatchDataset(train_samples, patch_size, patches_per_image,
                                              True, args.seed + stage_index * 10000)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"),
                            persistent_workers=(args.num_workers > 0))
        print(f"stage={stage_index} patch={patch_size} patches={patches_per_image} "
              f"batch={batch_size} steps_per_epoch={len(loader)}", flush=True)
        for local_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_loss = train_epoch(model, loader, optimizer, scaler, device, use_amp)
            eval_args = copy.copy(args)
            eval_args.patch_size = args.eval_patch_size
            eval_args.val_stride = args.eval_stride
            valid = evaluate(model, valid_samples, eval_args, device, use_amp)
            scheduler.step()
            selection = float(valid[args.selection_metric])
            record = {
                "stage": stage_index,
                "stage_patch_size": patch_size,
                "stage_local_epoch": local_epoch,
                "epoch": global_epoch,
                "train_loss": train_loss,
                "selection": selection,
                "valid": valid,
                "seconds": time.time() - start_time,
            }
            history.append(record)
            (output / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"stage": stage_index, "patch": patch_size,
                              "epoch": global_epoch, "train_loss": train_loss,
                              "selection": selection,
                              **{key: valid[key] for key in
                                 ("global_miou", "image_mean_miou",
                                  "image_mean_miou_exclude")}},
                             ensure_ascii=False), flush=True)
            if selection > best_score:
                best_score = selection
                best_record = record
                best_state = copy.deepcopy(model.state_dict())
                torch.save({"model": best_state, "epoch": global_epoch,
                            "metrics": valid, "stage": stage_index,
                            "stage_patch_size": patch_size, "args": vars(args)},
                           output / "best.pt")
        del loader, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_state is None:
        raise RuntimeError("no checkpoint was selected")
    model.load_state_dict(best_state)
    eval_args = copy.copy(args)
    eval_args.patch_size = args.eval_patch_size
    eval_args.val_stride = args.eval_stride
    final = evaluate(model, valid_samples, eval_args, device, use_amp, output_probs=True)
    probabilities = final.pop("probabilities")
    valid_payload = {path.stem: value.astype(np.float16)
                     for (path, _), value in zip(valid_samples, probabilities)}
    np.savez_compressed(output / "valid_probs.npz", **valid_payload)
    summary = {
        "selection_metric": args.selection_metric,
        "best_selection": best_score,
        "best_record": best_record,
        "final": final,
        "elapsed_seconds": time.time() - start_time,
        "representation": "raw_gray+per_image_zscore+empirical_cdf",
        "aspp_dilation": [1, 2, 4, 8],
        "stages": stages,
        "evaluation": {"patch_size": args.eval_patch_size, "stride": args.eval_stride},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--eval-patch-size", type=int, default=768)
    parser.add_argument("--eval-stride", type=int, default=576)
    parser.add_argument("--stage0-epochs", type=int, default=3)
    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=2)
    parser.add_argument("--stage0-patches", type=int, default=16)
    parser.add_argument("--stage1-patches", type=int, default=16)
    parser.add_argument("--stage2-patches", type=int, default=8)
    parser.add_argument("--stage0-batch", type=int, default=2)
    parser.add_argument("--stage1-batch", type=int, default=2)
    parser.add_argument("--stage2-batch", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--selection-metric", choices=("image_mean_miou",
                                                         "image_mean_miou_exclude"),
                        default="image_mean_miou_exclude")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
