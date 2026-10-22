import argparse
from alphase.data.audit import audit_dataset
from alphase.data.splits import make_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    report = audit_dataset(args.root, args.out)
    make_folds(f"{args.out}/manifest.csv")
    print(f"Prepared {report['counts']}; dataset SHA-256: {report['dataset_hash']}")


if __name__ == "__main__":
    main()
