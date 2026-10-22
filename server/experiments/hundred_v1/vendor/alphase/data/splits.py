import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from alphase.utils import sha256, write_json


def make_folds(manifest, folds=5, seed=2026):
    table = pd.read_csv(manifest).fillna("")
    labeled = table[table.source_split != "test"].reset_index(drop=True)
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result, summaries = [], []
    for k, (train, valid) in enumerate(splitter.split(labeled, groups=labeled.group_id)):
        a, b = labeled.iloc[train], labeled.iloc[valid]
        assert not set(a.group_id) & set(b.group_id)
        result.append(dict(fold=k, train_ids=a.image_id.tolist(), validation_ids=b.image_id.tolist()))
        for part, frame in (("train", a), ("valid", b)):
            summaries.append(dict(fold=k, split=part, images=len(frame), groups=frame.group_id.nunique(),
                                  absent_primary=int((frame.phase0_fraction == 0).sum()),
                                  mean_width=float(frame.width.mean()), mean_height=float(frame.height.mean()),
                                  **{f"phase{c}_fraction": float(frame[f"phase{c}_fraction"].mean()) for c in range(4)}))
    out = Path(manifest).parent
    write_json(out / "folds.json", dict(seed=seed, manifest_sha256=sha256(manifest), folds=result))
    pd.DataFrame(summaries).to_csv(out / "split_summary.csv", index=False)


def select_rows(cfg, split):
    table = pd.read_csv(cfg["data"]["manifest"]).fillna("")
    if split == "official":
        train, valid = table[table.source_split == "train"], table[table.source_split == "valid"]
    elif split == "all90":
        train, valid = table[table.source_split != "test"], table.iloc[:0]
    elif split.startswith("fold"):
        data = json.loads(Path(cfg["data"]["folds"]).read_text())
        if data["manifest_sha256"] != sha256(cfg["data"]["manifest"]):
            raise ValueError("Manifest changed since folds were generated")
        item = data["folds"][int(split.removeprefix("fold"))]
        train, valid = table[table.image_id.isin(item["train_ids"])], table[table.image_id.isin(item["validation_ids"])]
    else:
        raise ValueError("split must be official, all90 or fold0...fold4")
    if len(train) == 0 or (split != "all90" and len(valid) == 0):
        raise ValueError("Empty split")
    assert not set(train.image_id) & set(valid.image_id)
    return train.to_dict("records"), valid.to_dict("records")
