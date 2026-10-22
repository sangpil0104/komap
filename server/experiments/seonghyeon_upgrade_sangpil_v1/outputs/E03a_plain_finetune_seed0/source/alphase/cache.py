"""Native FP32 probability cache, atomic manifests, CPU streaming FP64 ensembles."""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from .contracts import CLASS_NAMES
from .config import inference_hash
from .io import (
    local_path,
    jsonl_read,
    jsonl_write,
    json_write,
    numpy_save,
    sha256,
    ids_hash,
    digest,
)


def check_probability(p, hw=None):
    if p.dtype != np.float32 or p.ndim != 3 or p.shape[0] != 4:
        raise ValueError("Expected float32 [4,H,W] probabilities")
    if hw is not None and list(p.shape[1:]) != list(hw):
        raise ValueError("Probability native shape mismatch")
    if not np.isfinite(p).all() or p.min() < -1e-6 or p.max() > 1 + 1e-6:
        raise ValueError("Invalid probability range/NaN")
    if not np.allclose(p.sum(0), 1, atol=1e-4, rtol=1e-4):
        raise ValueError("Probabilities do not sum to1")
    return p


def load_probability(row, root):
    for key, value in [
        ("schema_version", 1),
        ("class_names", list(CLASS_NAMES)),
        ("space", "native"),
        ("representation", "probability"),
        ("dtype", "float32"),
        ("completed", True),
    ]:
        if row.get(key) != value:
            raise ValueError(f"Cache metadata mismatch: {key}")
    for key in [
        "checkpoint_sha256",
        "image_sha256",
        "inference_config_hash",
        "training_ids_hash",
        "checkpoint_selection_ids_hash",
    ]:
        if not row.get(key) or row[key] == "UNKNOWN":
            raise ValueError(f"Unresolved cache provenance: {key}")
    path = Path(root) / row["probability_path"]
    if not path.is_file() or sha256(path) != row["probability_sha256"]:
        raise ValueError(f"Corrupt/missing cache: {path}")
    return check_probability(np.load(path, allow_pickle=False), row["native_hw"])


def cache_metadata(image, cfg, checkpoint_hash, train_ids, selection_ids):
    return dict(
        schema_version=1,
        image_id=image["image_id"],
        source_split=image["split"],
        native_hw=image["native_hw"],
        class_names=list(CLASS_NAMES),
        space="native",
        representation="probability",
        dtype="float32",
        checkpoint_sha256=checkpoint_hash,
        image_sha256=image["image_sha256"],
        inference_config_hash=inference_hash(cfg),
        training_ids_hash=ids_hash(train_ids),
        checkpoint_selection_ids_hash=ids_hash(selection_ids),
    )


def write_probability(root, metadata, prob):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    check_probability(prob, metadata["native_hw"])
    name = metadata["image_id"] + ".npy"
    path = root / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    numpy_save(path, prob)
    return {
        **metadata,
        "probability_path": name,
        "probability_sha256": sha256(path),
        "completed": True,
    }


def cache_model(
    cfg, checkpoint, images, out, device, train_ids, selection_ids, lineage_factory=None
):
    from .models import load_model
    from .inference import predict_probs
    from .data import verify_rows

    verify_rows(images)
    out = local_path(out)
    manifest = out / "manifest.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    rows = jsonl_read(manifest) if manifest.exists() else []
    existing = {r["image_id"]: r for r in rows}
    if len(existing) != len(rows) or not set(existing) <= {
        r["image_id"] for r in images
    }:
        raise ValueError("Cache image set mismatch")
    ckhash = sha256(local_path(checkpoint))
    model = None
    for image in tqdm(images, desc=f"cache {out.name}"):
        meta = cache_metadata(image, cfg, ckhash, train_ids, selection_ids)
        if lineage_factory:
            meta["lineage"] = lineage_factory(image)
        if image["image_id"] in existing:
            row = existing[image["image_id"]]
            if any(row.get(k) != v for k, v in meta.items()):
                raise ValueError("Stale cache config/checkpoint/lineage")
            load_probability(row, out)
            continue
        if model is None:
            model = load_model(cfg, checkpoint, device).eval()
        gray = np.array(Image.open(image["image_path"]).convert("L"))
        prob = predict_probs(model, gray, cfg, device).cpu().numpy().astype(np.float32)
        rows.append(write_probability(out, meta, prob))
        jsonl_write(manifest, rows)
    return manifest


def stream_mean(paths_and_rows):
    acc = None
    count = 0
    for root, row in paths_and_rows:
        p = load_probability(row, root)
        if acc is None:
            acc = np.zeros_like(p, dtype=np.float64)
        if acc.shape != p.shape:
            raise ValueError("Ensemble shape mismatch")
        acc += p
        count += 1
    if count == 0:
        raise ValueError("Empty ensemble")
    return (acc / count).astype(np.float32)


def ensemble(cfg, member_manifests=None, out=None, with_curve=True):
    e = cfg["ensemble"]
    out = local_path(out or e["output"])
    if member_manifests is None:
        member_manifests = [
            local_path(e["cache_root"]) / member / "manifest.jsonl"
            for member in e["fixed_member_ids"]
        ]
    if not member_manifests:
        raise ValueError("No members")
    lookup = []
    for path in member_manifests:
        path = Path(path)
        rows = jsonl_read(path)
        ids = {r["image_id"]: r for r in rows}
        if len(ids) != len(rows):
            raise ValueError("Duplicate cache image ID")
        lookup.append((path.parent, ids))
    keys = set(lookup[0][1])
    if any(set(ids) != keys for _, ids in lookup):
        raise ValueError("Missing member/image in ensemble")
    ensemble_hash = digest([sha256(p) for p in member_manifests])
    old = out / "manifest.jsonl"
    done = jsonl_read(old) if old.exists() else []
    completed = {r["image_id"]: r for r in done}
    curves = (
        {int(k): [] for k in e["k_curve"] if 0 < int(k) <= len(lookup)}
        if with_curve
        else {}
    )
    # Curves are independent predeclared prefixes, never validation-ranked members.
    for image_id in tqdm(sorted(keys), desc=f"ensemble K={len(lookup)}"):
        reference = lookup[0][1][image_id]
        for _, ids in lookup[1:]:
            row = ids[image_id]
            for field in [
                "image_id",
                "image_sha256",
                "native_hw",
                "class_names",
                "space",
                "representation",
                "source_split",
            ]:
                if row.get(field) != reference.get(field):
                    raise ValueError(f"Ensemble mismatch {field}")
        if image_id in completed:
            row = completed[image_id]
            if row.get("ensemble_hash") != ensemble_hash:
                raise ValueError("Changed ensemble definition")
            load_probability(row, out)
            # Recompute curve manifests if requested, otherwise validated resume.
            if not curves:
                continue
        acc = np.zeros((4, *reference["native_hw"]), np.float64)
        second = np.zeros_like(acc)
        votes = np.zeros_like(acc, dtype=np.uint16)
        ckhashes = []
        for i, (root, ids) in enumerate(lookup, 1):
            row = ids[image_id]
            p = load_probability(row, root)
            acc += p
            second += p.astype(np.float64) ** 2
            label = p.argmax(0)
            for c in range(4):
                votes[c] += label == c
            ckhashes.append(row["checkpoint_sha256"])
            if i in curves:
                prefix = out / f"prefix_{i:03d}"
                prefix.mkdir(parents=True, exist_ok=True)
                meta = {
                    **{
                        k: v
                        for k, v in reference.items()
                        if k
                        not in [
                            "probability_path",
                            "probability_sha256",
                            "completed",
                            "lineage",
                        ]
                    },
                    "checkpoint_sha256": digest(ckhashes),
                    "ensemble_hash": digest([sha256(p) for p in member_manifests[:i]]),
                    "inference_config_hash": digest(
                        [
                            ids[image_id]["inference_config_hash"]
                            for _, ids in lookup[:i]
                        ]
                    ),
                }
                mp = prefix / "manifest.jsonl"
                prev = {r["image_id"]: r for r in jsonl_read(mp)} if mp.exists() else {}
                if image_id in prev:
                    if prev[image_id]["ensemble_hash"] != meta["ensemble_hash"]:
                        raise ValueError("Changed prefix ensemble")
                    load_probability(prev[image_id], prefix)
                    r = prev[image_id]
                else:
                    r = write_probability(prefix, meta, (acc / i).astype(np.float32))
                prev[image_id] = r
                jsonl_write(mp, list(prev.values()))
        mean = (acc / len(lookup)).astype(np.float32)
        meta = {
            **{
                k: v
                for k, v in reference.items()
                if k
                not in [
                    "probability_path",
                    "probability_sha256",
                    "completed",
                    "lineage",
                ]
            },
            "checkpoint_sha256": digest(ckhashes),
            "ensemble_hash": ensemble_hash,
            "member_checkpoint_hashes": ckhashes,
            "inference_config_hash": digest(
                [ids[image_id]["inference_config_hash"] for _, ids in lookup]
            ),
            "selection_policy": e["selection"],
        }
        if image_id not in completed:
            done.append(write_probability(out, meta, mean))
            jsonl_write(old, done)
        variance = np.maximum(second / len(lookup) - (acc / len(lookup)) ** 2, 0)
        numpy_save(
            out / "diagnostics" / f"{image_id}_variance.npy",
            variance.astype(np.float32),
        )
        numpy_save(
            out / "diagnostics" / f"{image_id}_vote_fraction.npy",
            (votes / len(lookup)).astype(np.float32),
        )
    return old
