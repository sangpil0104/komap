"""Check actual losses on imbalanced labels, without models or optimizer steps."""
import ast
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/experiments"
HIST = SOURCE / "historical_scripts"
sys.path[:0] = [str(SOURCE), str(HIST), str(SOURCE / "model336_hrnet_best")]


@torch.inference_mode()
def main():
    torch.manual_seed(37)
    logits = torch.randn(2, 4, 8, 8)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, 0, :4] = 1
    target[:, 1, :3] = 2
    target[:, 2, :2] = 3
    plain_ce = F.cross_entropy(logits, target)
    weighted_ce = F.cross_entropy(logits, target, weight=torch.tensor([1.15, 1.45, .7, 2.]))
    assert not torch.isclose(plain_ce, weighted_ce)
    records = []
    cases = [
        ("improve_model6", "segmentation_loss", ("standard",)),
        ("improve_model6", "segmentation_loss", ("weighted",)),
        ("improve_five_causes", "patch_loss", ()),
        ("hard_negative_finetune", "loss_fn", ()),
        ("encoder_structure_search", "loss_fn", ()),
        ("resnet34_refinements", "segmentation_loss", ("ema", 1, 0)),
    ]
    for name, function, args in cases:
        module = importlib.import_module(name)
        result = getattr(module, function)(logits, target, *args)
        expected = plain_ce + module.dice_loss(logits, target)
        torch.testing.assert_close(result, expected)
        records.append({"module": name, "function": function, "args": args,
                        "loss": result.item(), "plain_ce_plus_dice": expected.item()})
    from encoder_structure_search import loss_fn
    from komap_seg.losses import ComboLoss
    combo = ComboLoss()
    torch.testing.assert_close(combo(logits, target), .5 * plain_ce + .5 * combo.dice(logits, target))
    assert combo.class_weights is None
    for weighted in (lambda: loss_fn(logits, target, torch.ones(4)),
                     lambda: ComboLoss(class_weights=torch.ones(4))):
        try:
            weighted()
        except ValueError as error:
            assert "disabled" in str(error)
        else:
            raise AssertionError("Class weights should not be re-enabled")
    from improve_model6 import CLASS_WEIGHTS
    from improve_five_causes import PATCH_CLASS_WEIGHTS
    from hard_negative_finetune import HARD_CLASS_WEIGHTS
    assert all(torch.equal(w, torch.ones(4)) for w in (CLASS_WEIGHTS, PATCH_CLASS_WEIGHTS, HARD_CLASS_WEIGHTS))

    # Inline CE in training loops is checked without executing those loops.
    modules = [HIST / (name + ".py") for name in (
        "model1_unet_resnet34", "train_models_2_to_6", "improve_model6", "improve_five_causes",
        "hard_negative_finetune", "encoder_structure_search", "pairwise_margin_finetune",
        "resnet34_refinements", "model181_raw_zcdf_aspp", "model194_multistage_team_schedule",
        "model239_native_stem_finetune", "model242_ema_continuation", "model253_highres_scale_finetune")]
    modules += [SOURCE / branch / "KoMaP/train.py" for branch in ("model322_team_b4", "model326_hybrid_fullstats")]
    modules += [SOURCE / "model336_hrnet_best/komap_seg/losses.py"]
    checked = []
    for path in modules:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name not in ("cross_entropy", "CrossEntropyLoss"):
                continue
            assert len(node.args) <= 2, (path, node.lineno, "positional weights")
            assert not any(k.arg == "weight" or k.arg is None for k in node.keywords), (path, node.lineno)
            checked.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert len(checked) >= 15
    report = {"training_started": False, "plain_ce": plain_ce.item(), "old_weighted_ce": weighted_ce.item(),
              "historical_losses": records, "hrnet_half_ce_half_dice": "passed",
              "class_weight_reenable_guard": "passed", "unweighted_ce_call_sites": checked}
    (ROOT / "reports/loss_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(cases)} historical losses, HRNet loss and {len(checked)} unweighted CE call sites")


if __name__ == "__main__":
    main()
