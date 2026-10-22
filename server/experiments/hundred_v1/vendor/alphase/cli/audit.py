import argparse
from pathlib import Path
from alphase.data.audit import audit_dataset
from alphase.utils import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--out", default="data/processed/audit.json")
    args = parser.parse_args()
    report = audit_dataset(args.root, Path(args.out).parent)
    write_json(args.out, report)
