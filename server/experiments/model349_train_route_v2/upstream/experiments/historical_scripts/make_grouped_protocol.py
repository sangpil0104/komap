"""Write a reproducible train-only specimen split without moving data."""
import argparse
import hashlib
import json
from pathlib import Path

from improve_five_causes import collect_split, specimen_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.data_dir)
    train = collect_split(root, 'train')
    official = collect_split(root, 'valid')
    groups = sorted({specimen_id(p) for p, _ in train},
                    key=lambda g: hashlib.sha256(('komap-177:' + g).encode()).hexdigest())
    held = set(groups[:max(1, round(len(groups) * .2))])
    def rows(samples):
        return [{'image': str(p.relative_to(root)), 'mask': str(m.relative_to(root)),
                 'specimen': specimen_id(p)} for p, m in samples]
    fit = [(p, m) for p, m in train if specimen_id(p) not in held]
    val = [(p, m) for p, m in train if specimen_id(p) in held]
    assert {specimen_id(p) for p, _ in fit}.isdisjoint(held)
    assert len(fit) + len(val) == len(train)
    result = {'fit': rows(fit), 'internal_valid': rows(val),
              'official_valid': rows(official),
              'checkpoint_requirement': 'Fresh initialization or external pretraining only; existing KoMaP checkpoints saw internal_valid.',
              'selection': 'Choose settings using internal_valid only; official_valid remains the requested final metric.',
              'official_specimen_overlap': sorted({specimen_id(p) for p, _ in train} &
                                                  {specimen_id(p) for p, _ in official})}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'fit': len(fit), 'internal_valid': len(val),
                      'held_specimens': len(held), 'output': str(output)}))


if __name__ == '__main__':
    main()
