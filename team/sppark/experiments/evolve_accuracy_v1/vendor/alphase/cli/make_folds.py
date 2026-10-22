import argparse
from alphase.data.splits import make_folds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--scheme", choices=["group"], default="group")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    make_folds(args.manifest, args.folds, args.seed)
