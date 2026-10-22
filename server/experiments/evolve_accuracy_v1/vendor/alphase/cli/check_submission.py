import argparse
from pathlib import Path
from PIL import Image
from alphase.data.palette import decode_mask


def check(inputs, predictions, expected_count=10):
    inputs = {p.name.removesuffix("_image.png"): p for p in Path(inputs).glob("*_image.png")}
    outputs = {p.name.removesuffix("_mask.png"): p for p in Path(predictions).glob("*_mask.png")}
    if set(inputs) != set(outputs) or len(inputs) != expected_count:
        raise ValueError("Input/output IDs or expected image count do not match")
    if len(list(Path(predictions).glob("*.png"))) != expected_count:
        raise ValueError("Unexpected PNG files in predictions directory")
    for key, path in inputs.items():
        with Image.open(path) as image:
            width, height = image.size
        with Image.open(outputs[key]) as mask:
            if mask.mode not in ("RGB", "RGBA"):
                raise ValueError(f"Expected RGB or RGBA output: {key}")
        if decode_mask(outputs[key]).shape != (height, width):
            raise ValueError(f"Output dimensions do not match: {key}")
    return len(inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="dataset/test/images")
    parser.add_argument("--predictions", default="submission/predictions")
    args = parser.parse_args()
    print(f"PASS: {check(args.inputs, args.predictions)} original-size palette masks")
