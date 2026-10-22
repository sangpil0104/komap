"""E3 학습 입력 만들기: train/valid 이미지를 기존 밝기·Z·CDF 3채널 RGB 8비트 PNG로 저장하고 마스크는 그대로 링크한다.
변환은 추론 때 쓰는 inference/e/zcdf_input.py의 encode()와 같다.
    python tools/make_zcdf_png.py --data DATA --out OUT"""
import argparse, os, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference' / 'e'))
from zcdf_input import encode

ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
for split in ('train', 'valid'):
    (a.out / split / 'images').mkdir(parents=True, exist_ok=True)
    for p in sorted((a.data / split / 'images').glob('*.png')):
        Image.fromarray(encode(p)).save(a.out / split / 'images' / p.name)
    m = a.out / split / 'masks'
    if not m.exists():
        os.symlink((a.data / split / 'masks').resolve(), m)
(a.out / 'done').write_text('ok\n')
print('zcdf png ->', a.out)
