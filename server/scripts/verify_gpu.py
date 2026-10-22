"""Real-resolution smoke tests and a small memorization diagnostic; no weights saved."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from alphase.data.dataset import build_inputs, channels, load_image, load_target
from alphase.evaluation.metrics import image_miou
from alphase.inference.predictor import predict_probabilities
from alphase.losses.segmentation import segmentation_loss
from alphase.models.factory import build_model
from alphase.utils import amp_context, load_config, seed_all, write_json


def main():
    cfg = load_config("configs/cnn_unet.yaml")
    torch.set_num_threads(4)
    seed_all(2026)
    device = torch.device("cuda")
    model = build_model(cfg).to(device)
    table = pd.read_csv(cfg["data"]["manifest"])
    report = {"shapes": []}
    for row in table.drop_duplicates(["width", "height"]).to_dict("records"):
        gray = load_image(row["image_path"]).astype(np.float32)/255
        prob = predict_probabilities(model, gray, cfg, device)
        assert prob.shape == (4, row["height"], row["width"])
        np.testing.assert_allclose(prob.sum(0), 1., atol=2e-6)
        report["shapes"].append(list(prob.shape))
        print("Native-resolution prediction passed:", prob.shape, flush=True)
    # Pick a real patch containing all four classes, without augmentation.
    selected = None
    for row in table[table.source_split == "train"].to_dict("records"):
        target = load_target(row["mask_path"])
        for y in range(0, target.shape[0]-127, 64):
            for x in range(0, target.shape[1]-127, 64):
                patch = target[y:y+128, x:x+128]
                if np.bincount(patch.ravel(), minlength=4).min() >= 200:
                    selected = (row, x, y, patch)
                    break
            if selected:
                break
        if selected:
            break
    if selected is None:
        raise ValueError("No four-class patch found")
    row, x, y, patch = selected
    gray = load_image(row["image_path"]).astype(np.float32)/255
    detail = torch.from_numpy(channels(gray)[:, y:y+128, x:x+128].copy())[None].to(device)
    target = torch.from_numpy(patch.astype(np.int64))[None].to(device)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    initial = None
    for step in range(250):
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, "bf16"):
            output = model(detail)
        loss, _ = segmentation_loss(output, target, cfg["loss"])
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step+1) % 50 == 0:
            print(f"Memorization diagnostic: {step+1}/250, loss={float(loss.detach()):.4f}", flush=True)
    with torch.no_grad(), amp_context(device, "bf16"):
        pred = model(detail)["logits"][0].argmax(0).cpu().numpy()
    score = image_miou(patch, pred)[0]
    report["memorization"] = dict(image_id=row["image_id"], xy=[x,y], size=128, updates=250,
                                  initial_loss=initial, final_loss=float(loss.detach()), training_patch_miou=score,
                                  note="Wiring diagnostic on one training patch; not a validation score.")
    assert float(loss.detach()) < initial*.25, report
    assert score > .90, report
    write_json("artifacts/gpu_verification.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
