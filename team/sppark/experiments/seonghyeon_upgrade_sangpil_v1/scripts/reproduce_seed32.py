import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from alphase.config import load_config
from alphase.io import ROOT, REPO, jsonl_read, json_write
from alphase.models import load_model
from alphase.legacy import native_probs, original_module
from alphase.contracts import (
    rgb_to_ids,
    confusion_matrix,
    metrics_from_cm,
    summarize_images,
)

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
cfg = load_config("configs/resolved_legacy.yaml")
model = load_model(cfg, cfg["paths"]["legacy_checkpoint"], "cuda:0").eval()
original = original_module(cfg)
rows = []
cms = []
for row in tqdm(
    [r for r in jsonl_read(ROOT / "manifests/dataset.jsonl") if r["split"] == "valid"],
    desc="seed32 full valid D4",
):
    gray = np.array(Image.open(row["image_path"]).convert("L"))
    prob = native_probs(model, gray, cfg, "cuda:0")
    target = rgb_to_ids(np.array(Image.open(row["mask_path"])))
    cm = confusion_matrix(target, prob.argmax(0).cpu().numpy())
    cms.append(cm)
    r = {
        "image_id": row["image_id"],
        **metrics_from_cm(cm),
        "confusion_matrix": cm.tolist(),
    }
    if row["image_id"] == "A71_step2_sample1":
        reference = original.predict_probs(
            [original.make_member(model.base, "imagenet", target_width=4000)],
            gray,
            tta=True,
            device="cuda:0",
        )
        r.update(
            max_abs_probability_difference=float((reference - prob).abs().max()),
            argmax_mismatch_pixels=int((reference.argmax(0) != prob.argmax(0)).sum()),
        )
        assert (
            r["max_abs_probability_difference"] == 0
            and r["argmax_mismatch_pixels"] == 0
        )
        np.save(ROOT / "reports/seed32_A71_probs.npy", prob.cpu().numpy())
    old = (
        REPO
        / "results/model_results_0926_2/predictions/seed_32"
        / Path(row["image_path"]).name
    )
    if old.exists():
        old_ids = rgb_to_ids(np.array(Image.open(old)))
        r["historical_mask_mismatch_pixels"] = int(
            (old_ids != prob.argmax(0).cpu().numpy()).sum()
        )
    rows.append(r)
result = {
    **summarize_images(cms),
    "historical_val_image_miou": 0.802265456654121,
    "per_image": rows,
}
result["historical_score_difference"] = (
    result["val_image_miou"] - result["historical_val_image_miou"]
)
json_write(ROOT / "reports/seed32_reproduction.json", result)
print({k: v for k, v in result.items() if k != "per_image"})
