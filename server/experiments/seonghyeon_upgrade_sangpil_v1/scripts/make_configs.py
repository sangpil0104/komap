import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
import yaml
from alphase.io import ROOT, jsonl_read
from alphase.config import DEFAULT, save_config

c = deepcopy(DEFAULT)
rows = jsonl_read(ROOT / "manifests/checkpoints.jsonl")
c["paths"]["legacy_checkpoint"] = rows[32]["checkpoint"]
save_config(ROOT / "configs/resolved_legacy.yaml", c)


def write(name, data):
    p = ROOT / "configs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False))


base = {
    "inherit": "../resolved_legacy.yaml",
    "run": {"name": "E02_scratch", "seed": 0},
    "model": {"family": "enhanced_b4_unet"},
    "geometry": {"profile": "native_roi_v1"},
    "sampling": {"image_selection": "uniform"},
    "training": {"augmentation": "native_shared_d4_photo"},
    "inference": {"mode": "dual_fov_native_tiles"},
}
write("experiments/E02_scratch.yaml", base)
write("experiments/E00.yaml", {"inherit": "../resolved_legacy.yaml"})
write(
    "experiments/E02.yaml",
    {
        "inherit": "E02_scratch.yaml",
        "run": {"name": "E02_geometry"},
        "paths": {"legacy_checkpoint": rows[0]["checkpoint"]},
    },
)
write(
    "experiments/E03a.yaml",
    {
        "inherit": "E02_scratch.yaml",
        "run": {"name": "E03a_plain_finetune"},
        "training": {
            "recipe": "finetune_existing",
            "initialize_from": rows[0]["checkpoint"],
            "total_optimizer_updates": 4000,
        },
    },
)
write(
    "experiments/E03b.yaml",
    {
        "inherit": "E03a.yaml",
        "run": {"name": "E03b_detail_residual"},
        "model": {"new_branches": {"residual_head": True}},
    },
)
for name, patch in {
    "E04": {"model": {"new_branches": {"texture": True}}},
    "E05": {"model": {"new_branches": {"context": True}}},
    "E06": {"sampling": {"random": 0.5, "mixed": 0.3, "hard": 0.2}},
    "E07": {"loss": {"native_enabled": True}},
    "E08_ema": {"training": {"ema_enabled": True}},
    "E08_bn": {"training": {"freeze_bn": True}},
}.items():
    write(
        f"experiments/{name}.yaml",
        {
            "inherit": "E03b.yaml" if name in ["E04", "E05"] else "E03a.yaml",
            "run": {"name": name},
            **patch,
        },
    )
write(
    "examples/enhanced_candidate.yaml",
    {
        "inherit": "../experiments/E03b.yaml",
        "run": {"name": "candidate_unvalidated"},
        "model": {"new_branches": {"texture": True, "context": True}},
        "sampling": {"random": 0.5, "mixed": 0.3, "hard": 0.2},
        "loss": {"native_enabled": True},
    },
)
write(
    "ensemble_uniform.yaml",
    {"inherit": "resolved_legacy.yaml", "run": {"name": "E01_uniform100"}},
)
write(
    "oof_train70.yaml",
    {"inherit": "resolved_legacy.yaml", "run": {"name": "strict_oof"}},
)
write(
    "refiner.yaml",
    {
        "inherit": "resolved_legacy.yaml",
        "run": {"name": "E10_refiner"},
        "refiner": {"validation_cache": "outputs/oof/valid/manifest.jsonl"},
    },
)
write(
    "final.yaml",
    {
        "inherit": "resolved_legacy.yaml",
        "run": {"name": "final_seed32_baseline"},
        "submission": {"checkpoint": rows[32]["checkpoint"]},
    },
)
