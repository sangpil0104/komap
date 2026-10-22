#!/usr/bin/env python3
"""Locked CPU-only Model397 fixed B4/HRNet grouped-OOF blend screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

import model396_fixed_architecture_oof as model396
import model392_widthnorm_oof as model392


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/model397_fixed_b4_model392_blend_preregistration.md"
SCRIPT = "experiments/model397_fixed_b4_model392_blend.py"
SMOKE = "experiments/model397_fixed_b4_model392_blend_smoke.py"
LOCK = "experiments/results/model397_fixed_b4_model392_blend_lock.json"
RESULT = "experiments/results/model397_fixed_b4_model392_blend_oof.json"
CLAIM = RESULT + ".claim"
MODEL392_LOCK = "experiments/results/model392_widthnorm_runtime_lock.json"
MODEL392_GATE = "experiments/results/model392_widthnorm_two_seed_confirmatory.json"
SEQUENCE_CLOSURE = "experiments/results/model392_sequence_closure_lock.json"
SEQUENCE_CLOSURE_SHA256 = "3df753de47ab636390ef806237018cd3c302636c0b59cc1dd2ea5236532a551e"
CV_ROOT = "data_model322_group_cv"
B4_ROOT = "outputs/model329_fullstats_width4000_uniform_group_oof"
MODEL392_ROOT = "outputs/model392_widthnorm"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 397
GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"_step(\d+)_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_record(root: Path) -> dict:
    path = root / MODEL392_GATE
    result = json.loads(path.read_text(encoding="utf-8"))
    lock_hash = sha256(root / MODEL392_LOCK)
    if (result.get("model") != 392
            or result.get("stage") != "two_seed_confirmatory"
            or result.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or result.get("model392_lock_sha256") != lock_hash
            or result.get("valid_evaluation_authorized")
            != result.get("summary", {}).get("gate_passed")):
        raise ValueError("Model392 two-seed gate result is missing or provenance-mismatched")
    return result


def input_paths(root: Path) -> list[Path]:
    masks = model396.mask_files(root)
    paths = [
        root / PROTOCOL,
        root / SCRIPT,
        root / SMOKE,
        root / model396.LOCK,
        root / MODEL392_LOCK,
        root / MODEL392_GATE,
        root / SEQUENCE_CLOSURE,
    ]
    for fold, fold_masks in masks.items():
        paths.extend(fold_masks.values())
        paths.extend((
            root / B4_ROOT / f"fold{fold}/args.json",
            root / B4_ROOT / f"fold{fold}_oof/probabilities.npz",
        ))
        for seed in (1, 2):
            run_dir = root / MODEL392_ROOT / f"seed{seed}/fold{fold}"
            oof_dir = root / MODEL392_ROOT / f"seed{seed}/fold{fold}_oof"
            paths.extend((
                run_dir / "last.pt",
                run_dir / "args.json",
                run_dir / "model392_run_receipt.json",
                oof_dir / "model392_manifest.json",
                oof_dir / "valid_probabilities.npz",
            ))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in paths})


def verify_model392_sequence_complete(root: Path, gate: dict) -> None:
    log_candidates = sorted(
        (root / "outputs").glob("model392_widthnorm_sequence_*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    if not log_candidates:
        raise FileNotFoundError("Model392 sequence completion log is missing")
    latest = log_candidates[-1]
    with latest.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 65_536), os.SEEK_SET)
        tail = stream.read().decode("utf-8", errors="replace")

    valid_dir = root / MODEL392_ROOT / "official_valid_once"
    if gate["summary"]["gate_passed"]:
        expected_end = "Model392 authorized one-shot official-valid evaluation completed; no test input was used."
        if expected_end not in tail:
            raise RuntimeError("Model392 gate passed, but its full-train/one-shot-valid sequence is not complete")
        required = (valid_dir / "valid_manifest.json", valid_dir / "valid_summary.json",
                    valid_dir / "valid_probabilities.npz", valid_dir / "valid_per_image.csv")
        if any(not path.is_file() for path in required):
            raise FileNotFoundError("Model392 one-shot valid artifacts are incomplete")
        manifest = json.loads((valid_dir / "valid_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("metric") != "image-wise 4-class mIoU; empty-union=0"
                or manifest.get("model392_lock_sha256") != sha256(root / MODEL392_LOCK)
                or manifest.get("oof_gate_sha256") != sha256(root / MODEL392_GATE)
                or len(manifest.get("valid_ids", [])) != 20
                or len(set(manifest.get("valid_ids", []))) != 20):
            raise ValueError("Model392 one-shot valid manifest is incomplete or provenance-mismatched")
    else:
        expected_end = "Gate failed; full-train and official-valid evaluation were not started."
        if expected_end not in tail:
            raise RuntimeError("Model392 OOF gate failed, but the runner did not record its clean no-valid exit")
        if valid_dir.exists():
            raise ValueError("Model392 official-valid artifacts exist although its locked OOF gate failed")


def verify_inputs(root: Path) -> dict:
    sequence_lock = root / MODEL392_ROOT / ".model392_oof.lock"
    if not sequence_lock.is_file():
        raise FileNotFoundError(f"Model392 sequence lock is missing: {sequence_lock}")
    with sequence_lock.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Model392 OOF/full-train/valid sequence is still active") from error
        finally:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    model396.verify_lock(root)
    model392.verify_model392_lock(root / MODEL392_LOCK)
    if sha256(root / SEQUENCE_CLOSURE) != SEQUENCE_CLOSURE_SHA256:
        raise ValueError("Model392 sequence-closure lock differs from its frozen hash")
    gate = gate_record(root)
    verify_model392_sequence_complete(root, gate)
    masks = model396.mask_files(root)
    b4_root = root / B4_ROOT
    for fold, fold_masks in masks.items():
        args = json.loads((b4_root / f"fold{fold}/args.json").read_text(encoding="utf-8"))
        if (any(args.get(key) != value for key, value in model396.B4_EXPECTED_ARGS.items())
                or not str(args.get("data", "")).endswith("/data_model322_group_cv")
                or args.get("train_splits") != f"train_fold{fold}"):
            raise ValueError(f"Model329 recipe/split mismatch in fold {fold}")
        expected_ids = set(fold_masks)
        with np.load(b4_root / f"fold{fold}_oof/probabilities.npz", allow_pickle=False) as archive:
            if set(archive.files) != expected_ids:
                raise ValueError(f"Model329 OOF IDs differ from fixed fold {fold}")
        for seed in (1, 2):
            run_dir = root / MODEL392_ROOT / f"seed{seed}/fold{fold}"
            oof_dir = root / MODEL392_ROOT / f"seed{seed}/fold{fold}_oof"
            archive_path = model392.verify_candidate_manifest(
                root, run_dir, oof_dir, seed, fold,
                root / MODEL392_LOCK, sha256(root / MODEL392_LOCK),
            )
            with np.load(archive_path, allow_pickle=False) as archive:
                if set(archive.files) != expected_ids:
                    raise ValueError(f"Model392 seed{seed} OOF IDs differ from fold {fold}")
    return {"model392_gate_passed": bool(gate["summary"]["gate_passed"]),
            "image_count": 70, "group_count": 59}


def create_lock(root: Path) -> dict:
    lock_path = root / LOCK
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite Model397 lock: {lock_path}")
    if (root / RESULT).exists() or (root / CLAIM).exists():
        raise FileExistsError("Model397 scoring was already claimed or published")
    verified = verify_inputs(root)
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths(root)
    }
    receipt = {
        "experiment": "Model397 fixed B4/Model392 grouped-OOF blend",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(model396.CLASS_NAMES),
        "folds": 4,
        "images": 70,
        "specimen_groups": 59,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model392_gate_passed_at_lock": verified["model392_gate_passed"],
        "valid_and_test_inputs_excluded": True,
        "locked_files": files,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def verify_lock(root: Path) -> dict:
    path = root / LOCK
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("experiment") != "Model397 fixed B4/Model392 grouped-OOF blend"
            or receipt.get("revision") != 1
            or receipt.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or receipt.get("bootstrap_draws") != BOOTSTRAP_DRAWS
            or receipt.get("bootstrap_seed") != BOOTSTRAP_SEED
            or receipt.get("valid_and_test_inputs_excluded") is not True):
        raise ValueError("unexpected Model397 lock/protocol settings")
    expected = {item.resolve().relative_to(root.resolve()).as_posix()
                for item in input_paths(root)}
    locked = receipt.get("locked_files", {})
    if set(locked) != expected:
        raise ValueError("Model397 input inventory differs from frozen lock")
    for name, metadata in locked.items():
        item = root / name
        if item.stat().st_size != metadata["size_bytes"] or sha256(item) != metadata["sha256"]:
            raise ValueError(f"Model397 locked input changed: {name}")
    return receipt


def blend_probabilities(b4: np.ndarray, seed1: np.ndarray,
                        seed2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = 0.5 * (seed1 + seed2)
    c = 0.5 * b4 + 0.25 * seed1 + 0.25 * seed2
    return b4, h, c


def gate(comparisons: dict) -> dict:
    against_h = comparisons["C_vs_H"]
    against_b4 = comparisons["C_vs_B4"]
    h_folds = against_h["fold_delta_pp"].values()
    checks = {
        "C_vs_H_gain_at_least_0_50pp": against_h["delta_image_mean_pp"] >= 0.50,
        "C_vs_H_group_CI_lower_above_zero": against_h["paired_group_bootstrap_95ci_pp"][0] > 0.0,
        "C_vs_H_positive_in_at_least_3_folds": against_h["positive_folds"] >= 3,
        "C_vs_H_worst_fold_at_least_minus_0_30pp": min(h_folds) >= -0.30,
        "C_vs_B4_gain_at_least_0_30pp": against_b4["delta_image_mean_pp"] >= 0.30,
        "C_vs_B4_group_CI_lower_above_zero": against_b4["paired_group_bootstrap_95ci_pp"][0] > 0.0,
        "C_vs_B4_positive_in_at_least_3_folds": against_b4["positive_folds"] >= 3,
        "classes_and_step4_safe_vs_H": (
            min(against_h["class_delta_pp"].values()) >= -0.50
            and against_h["step4_delta_pp"] >= -0.50
        ),
        "largest_group_gain_share_at_most_25_percent": (
            against_h["largest_group_positive_contribution_share"] is not None
            and against_h["largest_group_positive_contribution_share"] <= 0.25
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _claim(path: Path, lock_hash: str) -> str:
    record = {
        "experiment": "Model397 fixed B4/Model392 grouped-OOF blend",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_sha256": lock_hash,
        "policy": "one-shot; claim retained after interruption",
    }
    payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(path)


def score_oof(root: Path) -> dict:
    receipt = verify_lock(root)
    output_path = root / RESULT
    claim_path = root / CLAIM
    if output_path.exists() or claim_path.exists():
        raise FileExistsError("Model397 OOF scoring has already been claimed/published")
    claim_hash = _claim(claim_path, sha256(root / LOCK))
    masks = model396.mask_files(root)
    b4_root = root / B4_ROOT
    model392_root = root / MODEL392_ROOT
    rows: list[dict] = []
    scores = {name: [] for name in ("B4", "H", "C")}
    model392_lock = root / MODEL392_LOCK
    model392_lock_hash = sha256(model392_lock)
    for fold, mask_map in masks.items():
        ids = set(mask_map)
        with np.load(b4_root / f"fold{fold}_oof/probabilities.npz", allow_pickle=False) as b4_archive, \
             np.load(model392_root / f"seed1/fold{fold}_oof/valid_probabilities.npz", allow_pickle=False) as s1_archive, \
             np.load(model392_root / f"seed2/fold{fold}_oof/valid_probabilities.npz", allow_pickle=False) as s2_archive:
            if any(set(archive.files) != ids for archive in (b4_archive, s1_archive, s2_archive)):
                raise ValueError(f"OOF archive IDs differ in fixed fold {fold}")
            for image_id in sorted(ids):
                truth = model396.mask_to_ids(mask_map[image_id])
                p_b4 = np.asarray(b4_archive[image_id], dtype=np.float32)
                p_s1 = np.asarray(s1_archive[image_id], dtype=np.float32)
                p_s2 = np.asarray(s2_archive[image_id], dtype=np.float32)
                p_b4, p_h, p_c = blend_probabilities(p_b4, p_s1, p_s2)
                image_scores = {
                    "B4": model396.class_ious(p_b4, truth),
                    "H": model396.class_ious(p_h, truth),
                    "C": model396.class_ious(p_c, truth),
                }
                group = GROUP_RE.match(image_id)
                step = STEP_RE.search(image_id)
                if group is None or step is None:
                    raise ValueError(f"cannot parse specimen group/step from {image_id}")
                rows.append({"image": image_id, "group": group.group(1),
                             "step": int(step.group(1)), "fold": fold})
                for arm, value in image_scores.items():
                    scores[arm].append(value)
        print(f"validated and scored Model397 OOF fold {fold}/4", flush=True)
    if (len(rows) != 70 or len({row["image"] for row in rows}) != 70
            or len({row["group"] for row in rows}) != 59):
        raise ValueError("Model397 OOF coverage is not exactly 70 images / 59 groups")
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()}
    draws = model396.group_bootstrap_indices(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    comparisons = {
        "C_vs_H": model396.comparison(arrays["C"], arrays["H"], rows, draws),
        "C_vs_B4": model396.comparison(arrays["C"], arrays["B4"], rows, draws),
    }
    decision = gate(comparisons)
    model392_gate = gate_record(root)
    result = {
        "experiment": "Model397 fixed B4/Model392 grouped-OOF blend",
        "status": "OOF_SCREEN_PASS" if decision["pass"] else "NO_GO",
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(model396.CLASS_NAMES),
        "scope": "train-only exactly-once grouped OOF; selection-conditional development evidence",
        "official_valid_used": False,
        "test_used": False,
        "gpu_used": False,
        "lock_sha256": sha256(root / LOCK),
        "claim_sha256": claim_hash,
        "model392_primary_gate_passed": bool(model392_gate["summary"]["gate_passed"]),
        "valid_stage_allowed": bool(decision["pass"] and model392_gate["summary"]["gate_passed"]),
        "arms": {name: model396.overall(array) for name, array in arrays.items()},
        "comparisons": comparisons,
        "pre_registered_gate": decision,
        "input_snapshot_sha256": {
            "Model396_lock": sha256(root / model396.LOCK),
            "Model392_runtime_lock": model392_lock_hash,
            "Model392_primary_gate": sha256(root / MODEL392_GATE),
        },
        "decision_note": "OOF pass is only a screen; one additional valid look is allowed only when both this gate and Model392's primary OOF gate pass.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def self_test() -> None:
    truth = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    one_hot = np.zeros((4, 2, 2), dtype=np.float32)
    one_hot[0] = truth == 0
    one_hot[1] = truth == 1
    assert np.allclose(model396.class_ious(one_hot, truth), [1.0, 1.0, 0.0, 0.0])
    b4 = one_hot.copy()
    seed1 = one_hot.copy()
    seed2 = one_hot.copy()
    _, h, c = blend_probabilities(b4, seed1, seed2)
    assert np.array_equal(h, one_hot) and np.array_equal(c, one_hot)
    rows = [{"image": f"A{fold}_{group}_{n}", "group": f"A{fold}_{group}",
             "fold": fold, "step": 4}
            for fold in range(1, 5) for group in range(3) for n in range(2)]
    a = model396.group_bootstrap_indices(rows, 50, BOOTSTRAP_SEED)
    b = model396.group_bootstrap_indices(rows, 50, BOOTSTRAP_SEED)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(a) == 50
    mock_h = {
        "delta_image_mean_pp": 0.60,
        "paired_group_bootstrap_95ci_pp": [0.10, 0.90],
        "positive_folds": 3,
        "fold_delta_pp": {"1": -0.25, "2": 0.50, "3": 0.70, "4": 0.80},
        "class_delta_pp": {name: 0.05 for name in model396.CLASS_NAMES},
        "step4_delta_pp": 0.05,
        "largest_group_positive_contribution_share": 0.24,
    }
    mock_b4 = {
        "delta_image_mean_pp": 0.35,
        "paired_group_bootstrap_95ci_pp": [0.02, 0.70],
        "positive_folds": 3,
    }
    assert gate({"C_vs_H": mock_h, "C_vs_B4": mock_b4})["pass"]
    unsafe_h = {**mock_h, "class_delta_pp": {**mock_h["class_delta_pp"], "Al": -0.51}}
    assert not gate({"C_vs_H": unsafe_h, "C_vs_B4": mock_b4})["pass"]
    print("Model397 smoke passed: empty=0 metric, fixed blend arithmetic, deterministic fold-group bootstrap.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--create-lock", action="store_true")
    mode.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        self_test()
    elif args.create_lock:
        receipt = create_lock(root)
        print(json.dumps({"locked_files": len(receipt["locked_files"]),
                          "lock": str(root / LOCK)}, indent=2))
    elif args.verify_lock:
        receipt = verify_lock(root)
        print(json.dumps({"verified_files": len(receipt["locked_files"]),
                          "lock_sha256": sha256(root / LOCK)}, indent=2))
    else:
        print(json.dumps(score_oof(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
